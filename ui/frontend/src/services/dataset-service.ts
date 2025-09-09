/**
 * Dataset service for CRUD operations
 */

import { apiClient } from './api-client';
import { DatasetAPI, ApiResponse } from '../types/api';
import { Dataset, DatasetMetadata, CreateDatasetRequest, UpdateDatasetRequest } from '../types/dataset';
import { ErrorHandler } from '../utils/error-handler';

export class DatasetService implements DatasetAPI {
  /**
   * List all available datasets
   */
  async listDatasets(): Promise<ApiResponse<DatasetMetadata[]>> {
    try {
      const response = await apiClient.get<Dataset[]>('/datasets');
      
      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return {
          success: false,
          error: response.error,
          timestamp: new Date().toISOString(),
        };
      }

      // Convert full datasets to metadata
      const metadata: DatasetMetadata[] = response.data!.map(dataset => ({
        id: dataset.id,
        name: dataset.name,
        description: dataset.description,
        classCount: dataset.classes.length,
        embeddingsGenerated: dataset.embeddingsGenerated,
        createdAt: dataset.createdAt,
        updatedAt: dataset.updatedAt,
      }));

      return {
        success: true,
        data: metadata,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'filesystem',
          message: 'Failed to list datasets',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try again or check your connection.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Load a specific dataset by ID
   */
  async loadDataset(id: string): Promise<ApiResponse<Dataset>> {
    try {
      this.validateDatasetId(id);
      
      const response = await apiClient.get<Dataset>(`/datasets/${id}`);
      
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
          message: `Failed to load dataset ${id}`,
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check the dataset ID and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Save a dataset (create or update)
   */
  async saveDataset(dataset: CreateDatasetRequest | UpdateDatasetRequest): Promise<ApiResponse<Dataset>> {
    try {
      this.validateDatasetRequest(dataset);
      
      let response: ApiResponse<Dataset>;
      
      if ('id' in dataset) {
        // Update existing dataset
        response = await apiClient.put<Dataset>(`/datasets/${dataset.id}`, dataset);
      } else {
        // Create new dataset
        const createData = {
          ...dataset,
          id: this.generateDatasetId(dataset.name),
          embeddingsGenerated: false,
          createdAt: new Date(),
          updatedAt: new Date(),
        };
        
        response = await apiClient.post<Dataset>('/datasets', createData);
      }
      
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
          message: 'Failed to save dataset',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check your input and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Delete a dataset
   */
  async deleteDataset(id: string): Promise<ApiResponse<void>> {
    try {
      this.validateDatasetId(id);
      
      const response = await apiClient.delete<void>(`/datasets/${id}`);
      
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
          message: `Failed to delete dataset ${id}`,
          details: appError.message,
          recoverable: false,
          suggestedAction: 'Please check the dataset ID and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Generate a new dataset using domain-based generation
   */
  async generateDataset(domain: string, classCount: number): Promise<ApiResponse<Dataset>> {
    try {
      this.validateDomainGeneration(domain, classCount);
      
      const response = await apiClient.post<Dataset>('/wizard/generate-dataset', {
        domain,
        num_classes: classCount,
        examples_per_class: 3,
      });
      
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
          message: 'Failed to generate dataset',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try with a different domain or fewer classes.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Generate embeddings for a dataset
   */
  async generateEmbeddings(datasetId: string): Promise<ApiResponse<void>> {
    try {
      this.validateDatasetId(datasetId);
      
      const response = await apiClient.post<void>(`/datasets/${datasetId}/embeddings`);
      
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
          message: `Failed to generate embeddings for dataset ${datasetId}`,
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try again or check if the dataset exists.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Check if a dataset exists
   */
  async datasetExists(id: string): Promise<boolean> {
    try {
      const response = await this.loadDataset(id);
      return response.success;
    } catch {
      return false;
    }
  }

  /**
   * Get dataset statistics
   */
  async getDatasetStats(id: string): Promise<ApiResponse<{
    classCount: number;
    totalExamples: number;
    embeddingsGenerated: boolean;
    hasAttributes: boolean;
  }>> {
    try {
      const response = await this.loadDataset(id);
      
      if (!response.success) {
        return response as any;
      }

      const dataset = response.data!;
      const stats = {
        classCount: dataset.classes.length,
        totalExamples: dataset.classes.reduce((total, cls) => total + (cls.examples?.length || 0), 0),
        embeddingsGenerated: dataset.embeddingsGenerated,
        hasAttributes: false, // This would need to be checked via attribute service
      };

      return {
        success: true,
        data: stats,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'processing',
          message: `Failed to get stats for dataset ${id}`,
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check the dataset ID and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Validate dataset ID
   */
  private validateDatasetId(id: string): void {
    if (!id?.trim()) {
      throw ErrorHandler.createValidationError('Dataset ID is required', 'id');
    }

    if (id.length < 1 || id.length > 100) {
      throw ErrorHandler.createValidationError(
        'Dataset ID must be between 1 and 100 characters',
        'id'
      );
    }

    // Check for invalid characters
    if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
      throw ErrorHandler.createValidationError(
        'Dataset ID can only contain letters, numbers, underscores, and hyphens',
        'id'
      );
    }
  }

  /**
   * Validate dataset request
   */
  private validateDatasetRequest(dataset: CreateDatasetRequest | UpdateDatasetRequest): void {
    if (!dataset.name?.trim()) {
      throw ErrorHandler.createValidationError('Dataset name is required', 'name');
    }

    if (dataset.name.length < 1 || dataset.name.length > 100) {
      throw ErrorHandler.createValidationError(
        'Dataset name must be between 1 and 100 characters',
        'name'
      );
    }

    if (!dataset.description?.trim()) {
      throw ErrorHandler.createValidationError('Dataset description is required', 'description');
    }

    if (dataset.description.length > 500) {
      throw ErrorHandler.createValidationError(
        'Dataset description must be less than 500 characters',
        'description'
      );
    }

    if (!dataset.classes || dataset.classes.length === 0) {
      throw ErrorHandler.createValidationError('At least one class is required', 'classes');
    }

    if (dataset.classes.length > 50) {
      throw ErrorHandler.createValidationError(
        'Maximum 50 classes allowed per dataset',
        'classes'
      );
    }

    // Validate each class
    dataset.classes.forEach((cls, index) => {
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

      if (cls.name.length > 100) {
        throw ErrorHandler.createValidationError(
          `Class ${index + 1} name must be less than 100 characters`,
          `classes[${index}].name`
        );
      }

      if (cls.description.length > 500) {
        throw ErrorHandler.createValidationError(
          `Class ${index + 1} description must be less than 500 characters`,
          `classes[${index}].description`
        );
      }
    });

    // Check for duplicate class names
    const classNames = dataset.classes.map(cls => cls.name.toLowerCase());
    const duplicates = classNames.filter((name, index) => classNames.indexOf(name) !== index);
    
    if (duplicates.length > 0) {
      throw ErrorHandler.createValidationError(
        `Duplicate class names found: ${duplicates.join(', ')}`,
        'classes'
      );
    }
  }

  /**
   * Validate domain generation parameters
   */
  private validateDomainGeneration(domain: string, classCount: number): void {
    if (!domain?.trim()) {
      throw ErrorHandler.createValidationError('Domain is required', 'domain');
    }

    if (domain.length < 3 || domain.length > 100) {
      throw ErrorHandler.createValidationError(
        'Domain must be between 3 and 100 characters',
        'domain'
      );
    }

    if (!Number.isInteger(classCount) || classCount < 2 || classCount > 20) {
      throw ErrorHandler.createValidationError(
        'Class count must be an integer between 2 and 20',
        'classCount'
      );
    }
  }

  /**
   * Generate a unique dataset ID from name
   */
  private generateDatasetId(name: string): string {
    const cleanName = name
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, '_')
      .substring(0, 50);
    
    const timestamp = Date.now().toString(36);
    return `${cleanName}_${timestamp}`;
  }
}

// Default dataset service instance
export const datasetService = new DatasetService();