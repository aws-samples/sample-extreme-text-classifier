/**
 * Wizard service for domain-based dataset generation
 */

import { apiClient } from './api-client';
import { WizardAPI, ApiResponse } from '../types/api';
import { Dataset } from '../types/dataset';
import { ClassAttribute } from '../types/attributes';
import { ErrorHandler } from '../utils/error-handler';

export interface WizardGenerationOptions {
  classCount?: number;
  includeAttributes?: boolean;
  includeExamples?: boolean;
  examplesPerClass?: number;
}

export interface WizardGenerationResult {
  dataset: Dataset;
  attributes?: ClassAttribute[];
  examples?: string[];
}

export class WizardService implements WizardAPI {
  /**
   * Generate a complete dataset for a domain with classes, attributes, and examples
   */
  async generateDomainDataset(
    domain: string,
    options: WizardGenerationOptions = {}
  ): Promise<ApiResponse<WizardGenerationResult>> {
    try {
      this.validateDomainRequest(domain, options);

      const requestData = {
        domain,
        num_classes: options.classCount || 5,
        examples_per_class: options.examplesPerClass || 1,
        include_attributes: options.includeAttributes ?? true,
        include_examples: options.includeExamples ?? true,
      };

      const response = await apiClient.post<WizardGenerationResult>(
        '/wizard/generate-domain',
        requestData
      );

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
          type: 'processing',
          message: 'Domain dataset generation failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try with a different domain or fewer classes.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Generate classes for a specific domain
   */
  async generateDomainClasses(
    domain: string,
    classCount: number = 5
  ): Promise<ApiResponse<Dataset['classes']>> {
    try {
      this.validateDomain(domain);
      this.validateClassCount(classCount);

      const response = await apiClient.post<{ classes: Dataset['classes'] }>(
        '/wizard/generate-classes',
        {
          domain,
          num_classes: classCount,
        }
      );

      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return {
          success: false,
          error: response.error,
          timestamp: new Date().toISOString(),
        };
      }

      return {
        success: true,
        data: response.data!.classes,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'processing',
          message: 'Class generation failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try with a different domain.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Generate example documents for classes
   */
  async generateExamples(
    domain: string,
    classes: Dataset['classes'],
    examplesPerClass: number = 3
  ): Promise<ApiResponse<Record<string, string[]>>> {
    try {
      this.validateDomain(domain);
      this.validateClasses(classes);
      this.validateExamplesPerClass(examplesPerClass);

      const response = await apiClient.post<{ examples: Record<string, string[]> }>(
        '/wizard/generate-examples',
        {
          domain,
          classes: classes.map(cls => ({
            name: cls.name,
            description: cls.description,
          })),
          examples_per_class: examplesPerClass,
        }
      );

      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return {
          success: false,
          error: response.error,
          timestamp: new Date().toISOString(),
        };
      }

      return {
        success: true,
        data: response.data!.examples,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'processing',
          message: 'Example generation failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try with different classes or fewer examples.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Get domain suggestions based on input
   */
  async getDomainSuggestions(partialDomain: string): Promise<ApiResponse<string[]>> {
    try {
      if (!partialDomain?.trim()) {
        return {
          success: true,
          data: this.getDefaultDomainSuggestions(),
          timestamp: new Date().toISOString(),
        };
      }

      const response = await apiClient.get<{ suggestions: string[] }>(
        `/wizard/domain-suggestions?q=${encodeURIComponent(partialDomain)}`
      );

      if (!response.success) {
        // Fallback to default suggestions
        return {
          success: true,
          data: this.getDefaultDomainSuggestions(),
          timestamp: new Date().toISOString(),
        };
      }

      return {
        success: true,
        data: response.data!.suggestions,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      // Return default suggestions on error
      return {
        success: true,
        data: this.getDefaultDomainSuggestions(),
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Validate a domain for generation
   */
  async validateDomainForGeneration(domain: string): Promise<ApiResponse<{
    isValid: boolean;
    suggestions?: string[];
    estimatedClasses?: number;
  }>> {
    try {
      this.validateDomain(domain);

      const response = await apiClient.post<{
        is_valid: boolean;
        suggestions?: string[];
        estimated_classes?: number;
      }>('/wizard/validate-domain', { domain });

      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return response as any;
      }

      return {
        success: true,
        data: {
          isValid: response.data!.is_valid,
          suggestions: response.data!.suggestions,
          estimatedClasses: response.data!.estimated_classes,
        },
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'validation',
          message: 'Domain validation failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check your domain and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Get wizard progress for a generation request
   */
  async getGenerationProgress(requestId: string): Promise<ApiResponse<{
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    progress: number;
    currentStep: string;
    estimatedTimeRemaining?: number;
  }>> {
    try {
      if (!requestId?.trim()) {
        throw new Error('Request ID is required');
      }

      const response = await apiClient.get<{
        status: 'pending' | 'in_progress' | 'completed' | 'failed';
        progress: number;
        current_step: string;
        estimated_time_remaining?: number;
      }>(`/wizard/progress/${requestId}`);

      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return response as any;
      }

      return {
        success: true,
        data: {
          status: response.data!.status,
          progress: response.data!.progress,
          currentStep: response.data!.current_step,
          estimatedTimeRemaining: response.data!.estimated_time_remaining,
        },
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'processing',
          message: 'Failed to get generation progress',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check the request ID and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Get default domain suggestions
   */
  private getDefaultDomainSuggestions(): string[] {
    return [
      'medical supplies',
      'legal documents',
      'financial reports',
      'customer support tickets',
      'product reviews',
      'news articles',
      'academic papers',
      'business emails',
      'technical documentation',
      'marketing materials',
      'insurance claims',
      'real estate listings',
      'job applications',
      'restaurant reviews',
      'travel bookings',
      'software bugs',
      'scientific research',
      'educational content',
      'social media posts',
      'e-commerce products',
    ];
  }

  /**
   * Validate domain input
   */
  private validateDomain(domain: string): void {
    if (!domain?.trim()) {
      throw ErrorHandler.createValidationError('Domain is required', 'domain');
    }

    if (domain.length < 3) {
      throw ErrorHandler.createValidationError(
        'Domain must be at least 3 characters long',
        'domain'
      );
    }

    if (domain.length > 100) {
      throw ErrorHandler.createValidationError(
        'Domain must be less than 100 characters',
        'domain'
      );
    }

    // Check for reasonable domain format
    if (!/^[a-zA-Z0-9\s\-_.,]+$/.test(domain)) {
      throw ErrorHandler.createValidationError(
        'Domain contains invalid characters',
        'domain'
      );
    }
  }

  /**
   * Validate class count
   */
  private validateClassCount(classCount: number): void {
    if (!Number.isInteger(classCount) || classCount < 2 || classCount > 20) {
      throw ErrorHandler.createValidationError(
        'Class count must be an integer between 2 and 20',
        'classCount'
      );
    }
  }

  /**
   * Validate examples per class
   */
  private validateExamplesPerClass(examplesPerClass: number): void {
    if (!Number.isInteger(examplesPerClass) || examplesPerClass < 1 || examplesPerClass > 10) {
      throw ErrorHandler.createValidationError(
        'Examples per class must be an integer between 1 and 10',
        'examplesPerClass'
      );
    }
  }

  /**
   * Validate classes array
   */
  private validateClasses(classes: Dataset['classes']): void {
    if (!classes || classes.length === 0) {
      throw ErrorHandler.createValidationError('At least one class is required', 'classes');
    }

    if (classes.length > 20) {
      throw ErrorHandler.createValidationError(
        'Maximum 20 classes allowed',
        'classes'
      );
    }

    classes.forEach((cls, index) => {
      if (!cls.name?.trim()) {
        throw ErrorHandler.createValidationError(
          `Class ${index + 1} name is required`,
          `classes[${index}].name`
        );
      }

      if (!cls.description?.trim()) {
        throw ErrorHandler.createValidationError(
          `Class ${index + 1} description is required`,
          `classes[${index}].description`
        );
      }
    });
  }

  /**
   * Validate domain generation request
   */
  private validateDomainRequest(domain: string, options: WizardGenerationOptions): void {
    this.validateDomain(domain);

    if (options.classCount !== undefined) {
      this.validateClassCount(options.classCount);
    }

    if (options.examplesPerClass !== undefined) {
      this.validateExamplesPerClass(options.examplesPerClass);
    }
  }
}

// Default wizard service instance
export const wizardService = new WizardService();