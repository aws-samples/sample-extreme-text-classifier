/**
 * Attribute-related TypeScript interfaces and types
 */

export type AttributeConditionType = 'text_match' | 'numeric_range' | 'boolean' | 'custom';

export interface AttributeCondition {
  id: string;
  description: string;
  type: AttributeConditionType;
  parameters?: Record<string, any>;
}

export interface AttributeRule {
  operator: 'AND' | 'OR';
  conditions: (string | AttributeCondition | AttributeRule)[];
}

export interface ClassAttribute {
  classId: string;
  className: string;
  requiredAttributes: AttributeRule;
  generated: boolean;
  lastUpdated: Date;
}

export interface AttributeValidationResult {
  conditionsMet: string[];
  conditionsNotMet: string[];
  overallScore: number;
  details?: Record<string, any>;
  evaluationTree?: EvaluationTreeNode;
}

export interface EvaluationTreeNode {
  type: 'condition' | 'AND' | 'OR';
  satisfied: boolean | null;
  condition?: string;
  children?: EvaluationTreeNode[];
  skipped?: boolean;
  error?: string;
}

export interface GenerateAttributesRequest {
  datasetId: string;
  domain: string;
  classIds?: string[];
}

export interface SaveAttributesRequest {
  datasetId: string;
  attributes: ClassAttribute[];
}