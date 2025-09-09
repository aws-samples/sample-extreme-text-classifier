import React from 'react';
import { EvaluationTreeNode } from '../../types/attributes';

interface AttributeEvaluationTreeProps {
  tree: EvaluationTreeNode;
  level?: number;
}

export const AttributeEvaluationTree: React.FC<AttributeEvaluationTreeProps> = ({ 
  tree, 
  level = 0 
}) => {
  const getStatusIcon = (satisfied: boolean | null, skipped?: boolean) => {
    if (skipped) {
      return (
        <svg className="h-4 w-4 text-gray-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 012 0v4a1 1 0 11-2 0V7zM8 13a1 1 0 112 0 1 1 0 01-2 0z" clipRule="evenodd" />
        </svg>
      );
    }
    
    if (satisfied === true) {
      return (
        <svg className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      );
    } else if (satisfied === false) {
      return (
        <svg className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
      );
    } else {
      return (
        <svg className="h-4 w-4 text-gray-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
      );
    }
  };

  const getStatusColor = (satisfied: boolean | null, skipped?: boolean) => {
    if (skipped) return 'text-gray-600';
    if (satisfied === true) return 'text-green-700';
    if (satisfied === false) return 'text-red-700';
    return 'text-gray-600';
  };

  const getBackgroundColor = (satisfied: boolean | null, skipped?: boolean) => {
    if (skipped) return 'bg-gray-50 border-gray-200';
    if (satisfied === true) return 'bg-green-50 border-green-200';
    if (satisfied === false) return 'bg-red-50 border-red-200';
    return 'bg-gray-50 border-gray-200';
  };

  const getIndentClass = (level: number) => {
    switch (level) {
      case 0: return '';
      case 1: return 'ml-4';
      case 2: return 'ml-8';
      case 3: return 'ml-12';
      default: return 'ml-16';
    }
  };
  
  const indentClass = getIndentClass(level);

  if (tree.type === 'condition') {
    return (
      <div className={`${indentClass} mb-2`}>
        <div className={`p-3 rounded-lg border ${getBackgroundColor(tree.satisfied, tree.skipped)}`}>
          <div className="flex items-start space-x-2">
            {getStatusIcon(tree.satisfied, tree.skipped)}
            <div className="flex-1">
              <span className={`text-sm ${getStatusColor(tree.satisfied, tree.skipped)}`}>
                {tree.condition}
              </span>
              {tree.skipped && (
                <span className="text-xs text-gray-500 ml-2">(not evaluated)</span>
              )}
              {tree.error && (
                <div className="text-xs text-red-600 mt-1">
                  Error: {tree.error}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Operator node (AND/OR)
  return (
    <div className={`${indentClass} mb-3`}>
      <div className={`p-3 rounded-lg border ${getBackgroundColor(tree.satisfied)}`}>
        <div className="flex items-center space-x-2 mb-3">
          {getStatusIcon(tree.satisfied)}
          <span className={`px-2 py-1 rounded text-xs font-medium ${
            tree.type === 'AND' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'
          }`}>
            {tree.type}
          </span>
          <span className={`text-sm font-medium ${getStatusColor(tree.satisfied)}`}>
            {tree.type === 'AND' ? 'All conditions must be met' : 'At least one condition must be met'}
          </span>
          <span className="text-xs text-gray-500">
            ({tree.satisfied ? 'satisfied' : 'not satisfied'})
          </span>
        </div>
        
        <div className="space-y-2 pl-4 border-l-2 border-gray-200">
          {tree.children?.map((child, index) => (
            <AttributeEvaluationTree key={index} tree={child} level={level + 1} />
          ))}
        </div>
      </div>
    </div>
  );
};