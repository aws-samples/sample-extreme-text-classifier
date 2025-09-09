/**
 * Dataset File Service - Handles loading and saving datasets from/to files
 */

import { Dataset, DatasetMetadata } from '../types/dataset';
import { ErrorHandler } from '../utils/error-handler';

export class DatasetFileService {
  private static readonly DATASETS_PATH = '/datasets';

  /**
   * Load all available datasets from the datasets directory
   */
  static async loadAvailableDatasets(): Promise<Dataset[]> {
    try {
      const datasetFiles = ['medical_supplies.json', 'sample_documents.json'];
      const loadedDatasets: Dataset[] = [];

      for (const filename of datasetFiles) {
        try {
          const dataset = await this.loadDatasetFromFile(filename);
          if (dataset) {
            loadedDatasets.push(dataset);
          }
        } catch (error) {
          console.warn(`Failed to load dataset ${filename}:`, error);
        }
      }

      return loadedDatasets;
    } catch (error) {
      ErrorHandler.logError(error);
      throw new Error('Failed to load available datasets');
    }
  }

  /**
   * Load a specific dataset from a file
   */
  static async loadDatasetFromFile(filename: string): Promise<Dataset | null> {
    try {
      const response = await fetch(`${this.DATASETS_PATH}/${filename}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch dataset file: ${response.statusText}`);
      }

      const datasetData = await response.json();
      
      // Convert the file format to our Dataset interface
      const dataset: Dataset = {
        id: datasetData.id,
        name: datasetData.name,
        description: datasetData.description,
        classes: datasetData.classes.map((cls: any, index: number) => ({
          id: `${datasetData.id}_class_${index}`,
          name: cls.name,
          description: cls.description,
          examples: cls.examples || [],
          metadata: cls.metadata
        })),
        embeddingsGenerated: datasetData.embeddings_generated || false,
        createdAt: new Date(datasetData.created_at),
        updatedAt: new Date(datasetData.updated_at)
      };
      
      return dataset;
    } catch (error) {
      ErrorHandler.logError(error);
      return null;
    }
  }

  /**
   * Get metadata for all available datasets
   */
  static async getDatasetMetadata(): Promise<DatasetMetadata[]> {
    try {
      const datasets = await this.loadAvailableDatasets();
      return datasets.map(dataset => ({
        id: dataset.id,
        name: dataset.name,
        description: dataset.description,
        classCount: dataset.classes.length,
        embeddingsGenerated: dataset.embeddingsGenerated,
        createdAt: dataset.createdAt,
        updatedAt: dataset.updatedAt
      }));
    } catch (error) {
      ErrorHandler.logError(error);
      throw new Error('Failed to get dataset metadata');
    }
  }

  /**
   * Save a dataset to a file (for future implementation)
   * Note: This would require backend support for file writing
   */
  static async saveDatasetToFile(dataset: Dataset, filename?: string): Promise<void> {
    // This would require backend API support to write files
    // For now, we'll just log the operation
    console.log('Dataset save operation (requires backend support):', {
      dataset: dataset.name,
      filename: filename || `${dataset.id}.json`
    });
    
    // In a real implementation, this would make a POST request to save the file
    throw new Error('Dataset file saving requires backend API support');
  }

  /**
   * Delete a dataset file (for future implementation)
   * Note: This would require backend support for file deletion
   */
  static async deleteDatasetFile(datasetId: string): Promise<void> {
    // This would require backend API support to delete files
    console.log('Dataset delete operation (requires backend support):', datasetId);
    
    // In a real implementation, this would make a DELETE request
    throw new Error('Dataset file deletion requires backend API support');
  }
}