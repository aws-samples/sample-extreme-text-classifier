import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { ClassAttribute, AttributeRule } from '../../types/attributes';
import { ConditionBuilder } from './ConditionBuilder';

import { Save, X, AlertCircle, CheckCircle, Wand2 } from 'lucide-react';

interface AttributeEditorProps {
  attribute: ClassAttribute | null;
  isOpen: boolean;
  onSave: (attribute: ClassAttribute) => void;
  onCancel: () => void;
  onGenerate?: (classId: string, className: string) => void;
  isGenerating?: boolean;
  validationErrors?: string[];
}

export const AttributeEditor: React.FC<AttributeEditorProps> = ({
  attribute,
  isOpen,
  onSave,
  onCancel,
  onGenerate,
  isGenerating = false,
  validationErrors = []
}) => {
  const [editedAttribute, setEditedAttribute] = useState<ClassAttribute | null>(null);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (attribute) {
      setEditedAttribute({ ...attribute });
      setHasChanges(false);
    } else {
      setEditedAttribute(null);
      setHasChanges(false);
    }
  }, [attribute]);

  const handleSave = () => {
    if (editedAttribute) {
      const updatedAttribute: ClassAttribute = {
        ...editedAttribute,
        lastUpdated: new Date(),
        generated: false // Mark as manually edited
      };
      onSave(updatedAttribute);
    }
  };

  const handleCancel = () => {
    setEditedAttribute(null);
    setHasChanges(false);
    onCancel();
  };

  const handleGenerate = () => {
    if (editedAttribute && onGenerate) {
      onGenerate(editedAttribute.classId, editedAttribute.className);
    }
  };

  const updateAttribute = (updates: Partial<ClassAttribute>) => {
    if (editedAttribute) {
      setEditedAttribute({ ...editedAttribute, ...updates });
      setHasChanges(true);
    }
  };

  const updateRule = (rule: AttributeRule) => {
    updateAttribute({ requiredAttributes: rule });
  };

  const validateRule = (rule: AttributeRule): { isValid: boolean; errors: string[] } => {
    const errors: string[] = [];

    if (!rule.operator || !['AND', 'OR'].includes(rule.operator)) {
      errors.push('Rule operator must be either "AND" or "OR"');
    }

    if (!rule.conditions || rule.conditions.length === 0) {
      errors.push('Rule must have at least one condition');
    }

    // Add more validation logic as needed
    rule.conditions?.forEach((condition, index) => {
      if (typeof condition === 'string') {
        // String condition - basic validation
        const conditionStr = condition as string;
        if (!conditionStr.trim()) {
          errors.push(`Condition ${index + 1}: Cannot be empty`);
        }
      } else if (typeof condition === 'object' && condition !== null) {
        if ('operator' in condition) {
          // Nested rule validation
          const nestedValidation = validateRule(condition as AttributeRule);
          if (!nestedValidation.isValid) {
            errors.push(`Nested rule ${index + 1}: ${nestedValidation.errors.join(', ')}`);
          }
        } else {
          // Condition validation
          const cond = condition as any;
          if (!cond.id?.trim()) {
            errors.push(`Condition ${index + 1}: ID is required`);
          }
          if (!cond.description?.trim()) {
            errors.push(`Condition ${index + 1}: Description is required`);
          }
        }
      }
    });

    return {
      isValid: errors.length === 0,
      errors
    };
  };

  if (!isOpen || !editedAttribute) {
    return null;
  }

  const ruleValidation = validateRule(editedAttribute.requiredAttributes);
  const canSave = ruleValidation.isValid && hasChanges;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full h-[90vh] flex flex-col">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Edit Attributes: {editedAttribute.className}
              </h2>
              <p className="text-sm text-gray-500 mt-1">
                Define validation rules for enhanced classification accuracy
              </p>
            </div>
            <div className="flex items-center gap-2">
              {onGenerate && (
                <Button
                  variant="outline"
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="flex items-center"
                >
                  <Wand2 className={`w-4 h-4 mr-2 ${isGenerating ? 'animate-spin' : ''}`} />
                  {isGenerating ? 'Generating...' : 'Auto Generate'}
                </Button>
              )}
              <Button variant="ghost" onClick={handleCancel}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="space-y-6">
              {/* Basic Info */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Basic Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Class ID
                      </label>
                      <Input
                        value={editedAttribute.classId}
                        onChange={(e) => updateAttribute({ classId: e.target.value })}
                        placeholder="Unique class identifier"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Class Name
                      </label>
                      <Input
                        value={editedAttribute.className}
                        onChange={(e) => updateAttribute({ className: e.target.value })}
                        placeholder="Human-readable class name"
                      />
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <Badge 
                      variant={editedAttribute.generated ? "default" : "secondary"}
                      className="flex items-center"
                    >
                      {editedAttribute.generated ? (
                        <>
                          <Wand2 className="w-3 h-3 mr-1" />
                          Auto-Generated
                        </>
                      ) : (
                        <>
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Manually Edited
                        </>
                      )}
                    </Badge>
                    <span className="text-sm text-gray-500">
                      Last updated: {editedAttribute.lastUpdated.toLocaleDateString()}
                    </span>
                  </div>
                </CardContent>
              </Card>

              {/* Validation Errors */}
              {(validationErrors.length > 0 || !ruleValidation.isValid) && (
                <Card className="border-red-200 bg-red-50">
                  <CardContent className="pt-4">
                    <div className="flex items-start">
                      <AlertCircle className="w-5 h-5 text-red-600 mr-2 mt-0.5" />
                      <div>
                        <h4 className="text-sm font-medium text-red-800 mb-2">
                          Validation Errors
                        </h4>
                        <ul className="text-sm text-red-700 space-y-1">
                          {validationErrors.map((error, index) => (
                            <li key={index}>• {error}</li>
                          ))}
                          {ruleValidation.errors.map((error, index) => (
                            <li key={`rule-${index}`}>• {error}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Rule Builder */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Validation Rules
                </h3>
                <ConditionBuilder
                  rule={editedAttribute.requiredAttributes}
                  onChange={updateRule}
                  onValidate={validateRule}
                />
              </div>

              {/* Help Text */}
              <Card className="bg-blue-50 border-blue-200">
                <CardContent className="pt-4">
                  <h4 className="text-sm font-medium text-blue-800 mb-2">
                    How to Write Good Conditions
                  </h4>
                  <div className="text-sm text-blue-700 space-y-1">
                    <p>• <strong>Be specific:</strong> "Must contain a monetary amount to be paid" instead of "has money"</p>
                    <p>• <strong>Use keywords:</strong> "Contains terms like 'invoice', 'bill', or 'payment due'"</p>
                    <p>• <strong>Describe structure:</strong> "Has a table with itemized costs and quantities"</p>
                    <p>• <strong>AND rules:</strong> All conditions must be met for the class to match</p>
                    <p>• <strong>OR rules:</strong> At least one condition must be met for the class to match</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between p-6 border-t bg-gray-50">
            <div className="flex items-center text-sm text-gray-500">
              {hasChanges && (
                <span className="flex items-center">
                  <div className="w-2 h-2 bg-orange-400 rounded-full mr-2"></div>
                  Unsaved changes
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              <Button variant="outline" onClick={handleCancel}>
                Cancel
              </Button>
              <Button 
                onClick={handleSave} 
                disabled={!canSave}
                className="flex items-center"
              >
                <Save className="w-4 h-4 mr-2" />
                Save Changes
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};