import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { AttributeCondition, AttributeRule, AttributeConditionType } from '../../types/attributes';
import { Plus, Trash2, ChevronDown, ChevronRight } from 'lucide-react';

interface ConditionBuilderProps {
  rule: AttributeRule;
  onChange: (rule: AttributeRule) => void;
  onValidate?: (rule: AttributeRule) => { isValid: boolean; errors: string[] };
  className?: string;
  level?: number;
}

export const ConditionBuilder: React.FC<ConditionBuilderProps> = ({
  rule,
  onChange,
  onValidate,
  className = '',
  level = 0
}) => {
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set());

  // Auto-expand empty string conditions for immediate editing
  useEffect(() => {
    const emptyConditions = rule.conditions
      .map((condition, index) => ({ condition, index }))
      .filter(({ condition }) => typeof condition === 'string' && condition.trim() === '')
      .map(({ index }) => index);
    
    if (emptyConditions.length > 0) {
      setExpandedItems(prev => new Set([...Array.from(prev), ...emptyConditions]));
    }
  }, [rule.conditions]);

  const toggleExpanded = (index: number) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedItems(newExpanded);
  };

  const addCondition = () => {
    // Add a simple string condition
    const newIndex = rule.conditions.length;
    onChange({
      ...rule,
      conditions: [...rule.conditions, '']
    });
    // Auto-expand the new condition for immediate editing
    setExpandedItems(prev => new Set([...Array.from(prev), newIndex]));
  };

  const addNestedRule = (operator: 'AND' | 'OR' = 'AND') => {
    const newRule: AttributeRule = {
      operator,
      conditions: []
    };

    const newIndex = rule.conditions.length;
    onChange({
      ...rule,
      conditions: [...rule.conditions, newRule]
    });
    // Auto-expand the new nested rule
    setExpandedItems(prev => new Set([...Array.from(prev), newIndex]));
  };

  const updateCondition = (index: number, condition: string | AttributeCondition | AttributeRule) => {
    const newConditions = [...rule.conditions];
    newConditions[index] = condition;
    onChange({
      ...rule,
      conditions: newConditions
    });
  };

  const removeCondition = (index: number) => {
    onChange({
      ...rule,
      conditions: rule.conditions.filter((_, i) => i !== index)
    });
  };

  const toggleOperator = () => {
    onChange({
      ...rule,
      operator: rule.operator === 'AND' ? 'OR' : 'AND'
    });
  };

  const validation = onValidate ? onValidate(rule) : { isValid: true, errors: [] };

  return (
    <Card className={`${className} ${level > 0 ? 'border-l-4 border-l-blue-200 ml-4' : ''}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center">
            {level > 0 && <span className="text-gray-400 mr-2">└</span>}
            <Button
              variant="outline"
              size="sm"
              onClick={toggleOperator}
              className={`mr-2 ${rule.operator === 'AND' ? 'bg-blue-50 text-blue-700' : 'bg-orange-50 text-orange-700'}`}
            >
              {rule.operator}
            </Button>
            Rule ({rule.conditions.length} conditions)
          </CardTitle>
          <div className="flex items-center gap-1">
            <div className="relative">
              <select
                onChange={(e) => {
                  const value = e.target.value;
                  if (value === 'text') {
                    addCondition();
                  } else if (value === 'and') {
                    addNestedRule('AND');
                  } else if (value === 'or') {
                    addNestedRule('OR');
                  }
                  e.target.value = ''; // Reset selection
                }}
                className="h-7 px-2 text-xs border border-gray-300 rounded bg-white hover:bg-gray-50"
                defaultValue=""
              >
                <option value="" disabled>Add...</option>
                <option value="text">Text Condition</option>
                <option value="and">AND Group</option>
                <option value="or">OR Group</option>
              </select>
            </div>
          </div>
        </div>
        {!validation.isValid && (
          <div className="text-xs text-red-600 mt-1">
            {validation.errors.join(', ')}
          </div>
        )}
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-3">
          {rule.conditions.map((condition, index) => (
            <div key={index} className="relative">
              <div className="flex items-start gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => toggleExpanded(index)}
                  className="h-6 w-6 p-0 mt-1 border border-gray-300 hover:border-gray-400 text-xs font-bold"
                  title={expandedItems.has(index) ? "Collapse" : "Edit"}
                >
                  {expandedItems.has(index) ? "▼" : "✎"}
                </Button>
                
                <div className="flex-1">
                  {typeof condition === 'object' && condition !== null && 'operator' in condition ? (
                    // Nested rule
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant="secondary" className="text-xs">
                          Nested {condition.operator} Rule
                        </Badge>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeCondition(index)}
                          className="h-6 w-6 p-0 text-red-600 hover:text-red-700 font-bold"
                          title="Delete rule"
                        >
                          ×
                        </Button>
                      </div>
                      {expandedItems.has(index) && (
                        <ConditionBuilder
                          rule={condition as AttributeRule}
                          onChange={(newRule) => updateCondition(index, newRule)}
                          onValidate={onValidate}
                          level={level + 1}
                        />
                      )}
                    </div>
                  ) : (
                    // String condition (text condition)
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant="secondary" className="text-xs bg-blue-100 text-blue-800">
                          Text Condition
                        </Badge>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeCondition(index)}
                          className="h-6 w-6 p-0 text-red-600 hover:text-red-700 font-bold"
                          title="Delete condition"
                        >
                          ×
                        </Button>
                      </div>
                      {expandedItems.has(index) ? (
                        // Edit mode for string condition
                        <div className="space-y-2">
                          <Input
                            value={typeof condition === 'string' ? condition : ''}
                            onChange={(e) => updateCondition(index, e.target.value)}
                            placeholder="Must contain a monetary amount to be paid"
                            className="text-sm"
                          />
                        </div>
                      ) : (
                        // Display mode for string condition - clickable to edit
                        <div 
                          className="text-sm text-gray-700 bg-gray-50 rounded p-2 cursor-pointer hover:bg-gray-100"
                          onClick={() => toggleExpanded(index)}
                        >
                          {typeof condition === 'string' ? condition || 'Click to edit condition...' : 'Invalid condition'}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              
              {index < rule.conditions.length - 1 && (
                <div className="flex items-center justify-center my-2">
                  <Badge 
                    variant="secondary" 
                    className={`text-xs ${rule.operator === 'AND' ? 'bg-blue-50 text-blue-700' : 'bg-orange-50 text-orange-700'}`}
                  >
                    {rule.operator}
                  </Badge>
                </div>
              )}
            </div>
          ))}
          
          {rule.conditions.length === 0 && (
            <div className="text-center py-4 text-gray-500 text-sm">
              No conditions defined. Add a condition or nested rule to get started.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

interface ConditionEditorProps {
  condition: AttributeCondition;
  onChange: (condition: AttributeCondition) => void;
}

const ConditionEditor: React.FC<ConditionEditorProps> = ({ condition, onChange }) => {
  const updateCondition = (updates: Partial<AttributeCondition>) => {
    onChange({ ...condition, ...updates });
  };

  const updateParameters = (paramUpdates: Record<string, any>) => {
    onChange({
      ...condition,
      parameters: { ...condition.parameters, ...paramUpdates }
    });
  };

  return (
    <div className="space-y-3 p-3 bg-gray-50 rounded border">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            Condition Type
          </label>
          <select
            value={condition.type}
            onChange={(e) => updateCondition({ type: e.target.value as AttributeConditionType })}
            className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="text_match">Text Match</option>
            <option value="numeric_range">Numeric Range</option>
            <option value="boolean">Boolean</option>
            <option value="custom">Custom</option>
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            ID
          </label>
          <Input
            value={condition.id}
            onChange={(e) => updateCondition({ id: e.target.value })}
            className="text-sm"
            placeholder="Unique identifier"
          />
        </div>
      </div>
      
      <div>
        <label className="block text-xs font-medium text-gray-700 mb-1">
          Description
        </label>
        <Input
          value={condition.description}
          onChange={(e) => updateCondition({ description: e.target.value })}
          className="text-sm"
          placeholder="Describe what this condition checks for..."
        />
      </div>

      {condition.type === 'text_match' && (
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Match Pattern
            </label>
            <Input
              value={condition.parameters?.pattern || ''}
              onChange={(e) => updateParameters({ pattern: e.target.value })}
              className="text-sm"
              placeholder="Text pattern to match"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Case Sensitive
            </label>
            <select
              value={condition.parameters?.caseSensitive ? 'true' : 'false'}
              onChange={(e) => updateParameters({ caseSensitive: e.target.value === 'true' })}
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="false">No</option>
              <option value="true">Yes</option>
            </select>
          </div>
        </div>
      )}

      {condition.type === 'numeric_range' && (
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Minimum Value
            </label>
            <Input
              type="number"
              value={condition.parameters?.min || ''}
              onChange={(e) => updateParameters({ min: parseFloat(e.target.value) || undefined })}
              className="text-sm"
              placeholder="Min value"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Maximum Value
            </label>
            <Input
              type="number"
              value={condition.parameters?.max || ''}
              onChange={(e) => updateParameters({ max: parseFloat(e.target.value) || undefined })}
              className="text-sm"
              placeholder="Max value"
            />
          </div>
        </div>
      )}

      {condition.type === 'boolean' && (
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            Expected Value
          </label>
          <select
            value={condition.parameters?.expectedValue ? 'true' : 'false'}
            onChange={(e) => updateParameters({ expectedValue: e.target.value === 'true' })}
            className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        </div>
      )}

      {condition.type === 'custom' && (
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            Custom Logic
          </label>
          <textarea
            value={condition.parameters?.customLogic || ''}
            onChange={(e) => updateParameters({ customLogic: e.target.value })}
            className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            rows={3}
            placeholder="Define custom validation logic..."
          />
        </div>
      )}
    </div>
  );
};

const getConditionTypeColor = (type: AttributeConditionType): string => {
  switch (type) {
    case 'text_match':
      return 'bg-green-50 text-green-700';
    case 'numeric_range':
      return 'bg-blue-50 text-blue-700';
    case 'boolean':
      return 'bg-purple-50 text-purple-700';
    case 'custom':
      return 'bg-orange-50 text-orange-700';
    default:
      return 'bg-gray-50 text-gray-700';
  }
};

const getConditionTypeLabel = (type: AttributeConditionType): string => {
  switch (type) {
    case 'text_match':
      return 'Text Match';
    case 'numeric_range':
      return 'Numeric Range';
    case 'boolean':
      return 'Boolean';
    case 'custom':
      return 'Custom';
    default:
      return 'Unknown';
  }
};