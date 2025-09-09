import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { ClassAttribute } from '../../types/attributes';
import { Plus, Edit, Clock, CheckCircle, AlertTriangle, Timer } from 'lucide-react';

interface AttributeListProps {
  attributes: ClassAttribute[];
  isLoading?: boolean;
  onEdit: (attribute: ClassAttribute) => void;
  onDelete: (classId: string) => void;
  onGenerate: () => void;
  canGenerate?: boolean;
}

export const AttributeList: React.FC<AttributeListProps> = ({
  attributes,
  isLoading = false,
  onEdit,
  onDelete,
  onGenerate,
  canGenerate = true
}) => {
  const [showGenerationWarning, setShowGenerationWarning] = useState(false);





  const handleGenerateClick = () => {
    console.log('Generate button clicked!');
    
    // Create modal using the same method that worked in the test
    const testDiv = document.createElement('div');
    testDiv.innerHTML = `
      <div style="
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 999999;
        font-family: Arial, sans-serif;
      ">
        <div style="
          background-color: white;
          border-radius: 8px;
          padding: 24px;
          max-width: 400px;
          margin: 16px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
          <h3 style="color: #d97706; font-size: 18px; margin-bottom: 16px;">
            ⚠️ Confirm Attribute Generation
          </h3>
          <p style="color: #374151; font-size: 14px; margin-bottom: 20px; line-height: 1.5;">
            This process will take 3-5 minutes to complete. Please keep this tab open during processing.
          </p>
          <div style="display: flex; justify-content: flex-end; gap: 12px;">
            <button onclick="this.parentElement.parentElement.parentElement.parentElement.remove()" style="
              background-color: #f3f4f6;
              color: #374151;
              padding: 10px 16px;
              border: 1px solid #d1d5db;
              border-radius: 6px;
              font-size: 14px;
              cursor: pointer;
            ">Cancel</button>
            <button onclick="
              this.parentElement.parentElement.parentElement.parentElement.remove();
              window.startAttributeGeneration();
            " style="
              background-color: #2563eb;
              color: white;
              padding: 10px 16px;
              border: none;
              border-radius: 6px;
              font-size: 14px;
              cursor: pointer;
            ">Start Generation</button>
          </div>
        </div>
      </div>
    `;
    
    // Set up the global function for the onclick
    (window as any).startAttributeGeneration = () => {
      onGenerate();
    };
    
    document.body.appendChild(testDiv);
  };

  const handleConfirmGeneration = () => {
    console.log('Confirm generation clicked!');
    setShowGenerationWarning(false);
    onGenerate();
  };

  const handleCancelGeneration = () => {
    setShowGenerationWarning(false);
  };
  const formatConditionCount = (attribute: ClassAttribute): number => {
    const countConditions = (rule: any): number => {
      if (!rule || !rule.conditions) return 0;
      
      return rule.conditions.reduce((count: number, condition: any) => {
        if (condition.operator) {
          // Nested rule
          return count + countConditions(condition);
        } else {
          // Individual condition
          return count + 1;
        }
      }, 0);
    };
    
    try {
      return countConditions(attribute.requiredAttributes);
    } catch (error) {
      console.warn('Error counting conditions for attribute:', attribute, error);
      return 0;
    }
  };

  const safeGetOperator = (attribute: ClassAttribute): string => {
    return attribute.requiredAttributes?.operator || 'AND';
  };

  const safeGetConditions = (attribute: ClassAttribute): any[] => {
    return attribute.requiredAttributes?.conditions || [];
  };

  const isValidAttribute = (attribute: any): attribute is ClassAttribute => {
    return attribute && 
           typeof attribute.classId === 'string' && 
           typeof attribute.className === 'string' &&
           attribute.lastUpdated;
  };

  // Filter out any invalid attributes
  const validAttributes = attributes.filter(isValidAttribute);

  // Recursive function to render conditions with proper indentation
  const renderConditions = (conditions: any[], level: number = 0): React.ReactNode[] => {
    return conditions.map((condition, index) => {
      const indentClass = level > 0 ? `ml-${level * 4}` : '';
      const key = `${level}-${index}`;
      
      if (typeof condition === 'string') {
        // Text condition
        return (
          <div key={key} className={`text-xs text-gray-600 ${indentClass}`}>
            • {condition}
          </div>
        );
      } else if (typeof condition === 'object' && condition !== null && 'operator' in condition) {
        // Nested rule
        return (
          <div key={key} className={`text-xs ${indentClass}`}>
            <div className="text-gray-700 font-medium mb-1">
              {level > 0 ? '└ ' : ''}
              <span className="bg-gray-200 px-2 py-1 rounded text-xs">
                {condition.operator} Rule
              </span>
            </div>
            <div className="space-y-1">
              {renderConditions(condition.conditions || [], level + 1)}
            </div>
          </div>
        );
      } else {
        // Fallback for other condition types
        return (
          <div key={key} className={`text-xs text-gray-600 ${indentClass}`}>
            • {condition.description || 'Unknown condition'}
          </div>
        );
      }
    });
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <Card key={i} className="animate-pulse">
            <CardHeader>
              <div className="h-4 bg-gray-200 rounded w-1/3"></div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="h-3 bg-gray-200 rounded w-full"></div>
                <div className="h-3 bg-gray-200 rounded w-2/3"></div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (validAttributes.length === 0) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto">
              <Plus className="w-8 h-8 text-gray-400" />
            </div>
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                No Attributes Found
              </h3>
              <p className="text-gray-500 mb-4">
                Generate attributes to enhance classification accuracy with validation rules.
              </p>
              {canGenerate && (
                <Button onClick={handleGenerateClick} className="inline-flex items-center">
                  <Plus className="w-4 h-4 mr-2" />
                  Generate Attributes
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">

      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-medium text-gray-900">
            Class Attributes ({validAttributes.length})
          </h3>
          <p className="text-sm text-gray-500">
            Validation rules for enhanced classification accuracy
          </p>
        </div>
        {canGenerate && (
          <Button onClick={handleGenerateClick} variant="outline" size="sm">
            <Plus className="w-4 h-4 mr-2" />
            Regenerate All
          </Button>
        )}
      </div>

      <div className="grid gap-4">
        {validAttributes.map((attribute) => (
          <Card key={attribute.classId} className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <CardTitle className="text-base font-medium text-gray-900">
                    {attribute.className}
                  </CardTitle>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge 
                      variant={attribute.generated ? "default" : "secondary"}
                      className="text-xs"
                    >
                      {attribute.generated ? (
                        <>
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Generated
                        </>
                      ) : (
                        <>
                          <Edit className="w-3 h-3 mr-1" />
                          Manual
                        </>
                      )}
                    </Badge>
                    <span className="text-xs text-gray-500 flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      {attribute.lastUpdated.toLocaleDateString()}
                    </span>
                  </div>
                </div>
                <div className="flex items-center">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onEdit(attribute)}
                    className="h-9 px-3"
                  >
                    <Edit className="w-4 h-4 mr-2" />
                    Edit
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Validation Rules:</span>
                  <Badge variant="secondary" className="text-xs">
                    {formatConditionCount(attribute)} conditions
                  </Badge>
                </div>
                <div className="text-sm text-gray-700 bg-gray-50 rounded p-2">
                  <div className="font-medium mb-1">
                    {safeGetOperator(attribute)} Rule:
                  </div>
                  <div className="space-y-1">
                    {renderConditions(safeGetConditions(attribute), 0)}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Generation Warning Dialog - SUPER SIMPLE TEST */}
      {showGenerationWarning && createPortal(
        <div style={{
          position: 'fixed',
          top: '50px',
          left: '50px',
          width: '300px',
          height: '200px',
          backgroundColor: 'red',
          color: 'white',
          zIndex: 999999,
          border: '5px solid yellow',
          fontSize: '20px',
          padding: '20px'
        }}>
          <div>MODAL TEST - CAN YOU SEE THIS?</div>
          <button 
            onClick={handleCancelGeneration}
            style={{ 
              backgroundColor: 'blue', 
              color: 'white', 
              padding: '10px', 
              margin: '10px',
              fontSize: '16px'
            }}
          >
            Cancel
          </button>
          <button 
            onClick={handleConfirmGeneration}
            style={{ 
              backgroundColor: 'green', 
              color: 'white', 
              padding: '10px', 
              margin: '10px',
              fontSize: '16px'
            }}
          >
            Start Generation
          </button>
        </div>,
        document.body
      )}
    </div>
  );
};