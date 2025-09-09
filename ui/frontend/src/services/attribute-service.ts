/**
 * Attribute service for generation and management
 */

import { apiClient } from './api-client';
import { AttributeAPI, ApiResponse } from '../types/api';
import { 
  ClassAttribute, 
  GenerateAttributesRequest, 
  SaveAttributesRequest,
  AttributeRule,
  AttributeCondition
} from '../types/attributes';
import { ErrorHandler } from '../utils/error-handler';

export class AttributeService implements AttributeAPI {
  /**
   * Get attributes for a dataset
   */
  async getAttributes(datasetId: string): Promise<ApiResponse<ClassAttribute[]>> {
    try {
      this.validateDatasetId(datasetId);
      
      const response = await apiClient.get<{
        metadata: {
          dataset_id: string;
          domain: string;
          generated: boolean;
          last_updated: string;
        };
        classes: Array<{
          name: string;
          description: string;
          required_attributes: any;
        }>;
      }>(`/attributes/${datasetId}`);
      
      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return {
          success: false,
          error: response.error,
          timestamp: new Date().toISOString(),
        };
      }

      // Convert API response to ClassAttribute array
      console.log('Raw API response:', response.data);
      
      const metadata = response.data?.metadata || { dataset_id: datasetId, domain: '', generated: false, last_updated: new Date().toISOString() };
      const classes = response.data?.classes || [];
      
      console.log('Metadata:', metadata);
      console.log('Classes:', classes);
      
      // Convert classes to ClassAttribute format
      const processedAttributes = classes.map((cls) => ({
        classId: cls.name.toLowerCase().replace(/\s+/g, '_'),
        className: cls.name,
        requiredAttributes: cls.required_attributes || {
          operator: 'AND' as const,
          conditions: []
        },
        generated: metadata.generated,
        lastUpdated: new Date(metadata.last_updated),
      } as ClassAttribute));
      
      console.log('Processed attributes:', processedAttributes);

      return {
        success: true,
        data: processedAttributes,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'filesystem',
          message: `Failed to get attributes for dataset ${datasetId}`,
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check the dataset ID and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Generate attributes for a dataset using LLM
   */
  async generateAttributes(request: GenerateAttributesRequest): Promise<ApiResponse<ClassAttribute[]>> {
    try {
      this.validateGenerateRequest(request);
      
      const formData = new FormData();
      formData.append('domain', request.domain);
      
      if (request.classIds && request.classIds.length > 0) {
        formData.append('class_ids', JSON.stringify(request.classIds));
      }

      const response = await apiClient.request<{
        metadata: {
          dataset_id: string;
          domain: string;
          generated: boolean;
          last_updated: string;
        };
        classes: Array<{
          name: string;
          description: string;
          required_attributes: any;
        }>;
      }>(`/attributes/${request.datasetId}/generate`, {
        method: 'POST',
        body: formData,
        headers: {}, // Let browser set Content-Type for FormData
        timeout: 300000, // 5 minutes timeout for attribute generation
      });
      
      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return {
          success: false,
          error: response.error,
          timestamp: new Date().toISOString(),
        };
      }

      // Convert API response to ClassAttribute array
      const metadata = response.data?.metadata || { dataset_id: request.datasetId, domain: request.domain, generated: true, last_updated: new Date().toISOString() };
      const classes = response.data?.classes || [];
      
      const processedAttributes = classes.map((cls: any) => ({
        classId: cls.name.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, ''),
        className: cls.name,
        requiredAttributes: cls.required_attributes || {
          operator: 'AND' as const,
          conditions: []
        },
        generated: metadata.generated,
        lastUpdated: new Date(metadata.last_updated),
      } as ClassAttribute));

      console.log('Processed attributes:', processedAttributes);

      return {
        success: true,
        data: processedAttributes,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'processing',
          message: 'Failed to generate attributes',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try with a different domain or check your dataset.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Save attributes for a dataset
   */
  async saveAttributes(request: SaveAttributesRequest): Promise<ApiResponse<void>> {
    try {
      this.validateSaveRequest(request);
      
      const requestData = {
        metadata: {
          dataset_id: request.datasetId,
          domain: "User edited attributes",
          generated: false, // User edited
          last_updated: new Date().toISOString(),
        },
        classes: request.attributes.map(attr => ({
          name: attr.className,
          description: `${attr.className} class attributes`,
          required_attributes: attr.requiredAttributes,
        })),
      };

      const response = await apiClient.put<void>(`/attributes/${request.datasetId}`, requestData);
      
      if (!response.success) {
        ErrorHandler.logError(response.error!);
      }

      return response;
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'filesystem',
          message: 'Failed to save attributes',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check your input and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Check if attributes exist for a dataset
   */
  async attributesExist(datasetId: string): Promise<boolean> {
    try {
      const response = await this.getAttributes(datasetId);
      return response.success && response.data!.length > 0;
    } catch {
      return false;
    }
  }

  /**
   * Delete attributes for a dataset
   */
  async deleteAttributes(datasetId: string): Promise<ApiResponse<void>> {
    try {
      this.validateDatasetId(datasetId);
      
      const response = await apiClient.delete<void>(`/attributes/${datasetId}`);
      
      if (!response.success) {
        ErrorHandler.logError(response.error!);
      }

      return response;
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'filesystem',
          message: `Failed to delete attributes for dataset ${datasetId}`,
          details: appError.message,
          recoverable: false,
          suggestedAction: 'Please check the dataset ID and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Validate an attribute rule structure
   */
  validateAttributeRule(rule: AttributeRule): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!rule.operator || !['AND', 'OR'].includes(rule.operator)) {
      errors.push('Rule operator must be either "AND" or "OR"');
    }

    if (!rule.conditions || rule.conditions.length === 0) {
      errors.push('Rule must have at least one condition');
    }

    if (rule.conditions) {
      rule.conditions.forEach((condition, index) => {
        if (typeof condition === 'string') {
          // String condition - basic validation
          if (!condition.trim()) {
            errors.push(`Condition ${index + 1}: Cannot be empty`);
          }
        } else if (typeof condition === 'object' && condition !== null) {
          if ('operator' in condition) {
            // Nested rule
            const nestedValidation = this.validateAttributeRule(condition as AttributeRule);
            if (!nestedValidation.isValid) {
              errors.push(`Nested rule ${index + 1}: ${nestedValidation.errors.join(', ')}`);
            }
          } else {
            // AttributeCondition object validation
            const conditionValidation = this.validateAttributeCondition(condition as AttributeCondition);
            if (!conditionValidation.isValid) {
              errors.push(`Condition ${index + 1}: ${conditionValidation.errors.join(', ')}`);
            }
          }
        }
      });
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate an attribute condition
   */
  validateAttributeCondition(condition: AttributeCondition): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!condition.id?.trim()) {
      errors.push('Condition ID is required');
    }

    if (!condition.description?.trim()) {
      errors.push('Condition description is required');
    }

    if (!condition.type || !['text_match', 'numeric_range', 'boolean', 'custom'].includes(condition.type)) {
      errors.push('Condition type must be one of: text_match, numeric_range, boolean, custom');
    }

    if (condition.description && condition.description.length > 200) {
      errors.push('Condition description must be less than 200 characters');
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Create a simple text match condition
   */
  createTextMatchCondition(description: string, parameters: Record<string, any> = {}): AttributeCondition {
    return {
      id: this.generateConditionId(),
      description,
      type: 'text_match',
      parameters,
    };
  }

  /**
   * Create a numeric range condition
   */
  createNumericRangeCondition(
    description: string,
    min?: number,
    max?: number,
    parameters: Record<string, any> = {}
  ): AttributeCondition {
    return {
      id: this.generateConditionId(),
      description,
      type: 'numeric_range',
      parameters: {
        ...parameters,
        min,
        max,
      },
    };
  }

  /**
   * Create a boolean condition
   */
  createBooleanCondition(
    description: string,
    expectedValue: boolean,
    parameters: Record<string, any> = {}
  ): AttributeCondition {
    return {
      id: this.generateConditionId(),
      description,
      type: 'boolean',
      parameters: {
        ...parameters,
        expectedValue,
      },
    };
  }

  /**
   * Create a simple AND rule
   */
  createAndRule(conditions: (AttributeCondition | AttributeRule)[]): AttributeRule {
    return {
      operator: 'AND',
      conditions,
    };
  }

  /**
   * Create a simple OR rule
   */
  createOrRule(conditions: (AttributeCondition | AttributeRule)[]): AttributeRule {
    return {
      operator: 'OR',
      conditions,
    };
  }

  /**
   * Validate dataset ID
   */
  private validateDatasetId(datasetId: string): void {
    if (!datasetId?.trim()) {
      throw ErrorHandler.createValidationError('Dataset ID is required', 'datasetId');
    }
  }

  /**
   * Validate generate attributes request
   */
  private validateGenerateRequest(request: GenerateAttributesRequest): void {
    this.validateDatasetId(request.datasetId);

    if (!request.domain?.trim()) {
      throw ErrorHandler.createValidationError('Domain is required', 'domain');
    }

    if (request.domain.length < 3 || request.domain.length > 100) {
      throw ErrorHandler.createValidationError(
        'Domain must be between 3 and 100 characters',
        'domain'
      );
    }

    if (request.classIds && request.classIds.length > 50) {
      throw ErrorHandler.createValidationError(
        'Maximum 50 class IDs allowed',
        'classIds'
      );
    }
  }

  /**
   * Validate save attributes request
   */
  private validateSaveRequest(request: SaveAttributesRequest): void {
    this.validateDatasetId(request.datasetId);

    if (!request.attributes || request.attributes.length === 0) {
      throw ErrorHandler.createValidationError('At least one attribute is required', 'attributes');
    }

    if (request.attributes.length > 50) {
      throw ErrorHandler.createValidationError(
        'Maximum 50 attributes allowed',
        'attributes'
      );
    }

    // Validate each attribute
    request.attributes.forEach((attr, index) => {
      if (!attr.classId?.trim()) {
        throw ErrorHandler.createValidationError(
          `Attribute ${index + 1} class ID is required`,
          `attributes[${index}].classId`
        );
      }

      if (!attr.className?.trim()) {
        throw ErrorHandler.createValidationError(
          `Attribute ${index + 1} class name is required`,
          `attributes[${index}].className`
        );
      }

      if (!attr.requiredAttributes) {
        throw ErrorHandler.createValidationError(
          `Attribute ${index + 1} required attributes are required`,
          `attributes[${index}].requiredAttributes`
        );
      }

      // Validate the attribute rule
      const ruleValidation = this.validateAttributeRule(attr.requiredAttributes);
      if (!ruleValidation.isValid) {
        throw ErrorHandler.createValidationError(
          `Attribute ${index + 1} rule validation failed: ${ruleValidation.errors.join(', ')}`,
          `attributes[${index}].requiredAttributes`
        );
      }
    });
  }

  /**
   * Generate a unique condition ID
   */
  private generateConditionId(): string {
    return `condition_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }
}

// Default attribute service instance
export const attributeService = new AttributeService();