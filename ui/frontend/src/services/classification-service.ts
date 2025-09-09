/**
 * Classification service with support for different options
 */

import { apiClient } from './api-client';
import {
  ClassificationAPI,
  ApiResponse
} from '../types/api';
import {
  ClassificationResult,
  ClassifyTextRequest,
  ClassifyPDFRequest,
  PDFExtractionResult,
  ClassificationConfig
} from '../types/classification';
import { ErrorHandler } from '../utils/error-handler';

export class ClassificationService implements ClassificationAPI {
  /**
   * Classify text using the specified dataset and configuration
   */
  async classifyText(request: ClassifyTextRequest): Promise<ApiResponse<ClassificationResult>> {
    try {
      this.validateTextRequest(request);

      const response = await apiClient.post<any>('/classify/text', {
        text: request.text,
        dataset_id: request.datasetId,
        config: {
          use_reranking: request.config.useReranking,
          reranking_model: request.config.rerankingModel || 'us.amazon.nova-lite-v1:0',
          use_attribute_validation: request.config.useAttributeValidation,
          top_k_candidates: request.config.topKCandidates,
        },
      });

      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return response;
      }

      // Transform backend response to frontend format
      const backendResult = response.data;
      console.log('Backend classification result:', backendResult); // Debug log

      const classificationResult: ClassificationResult = {
        predictedClass: {
          id: backendResult.predicted_class.name.toLowerCase().replace(/\s+/g, '_'),
          name: backendResult.predicted_class.name,
          description: backendResult.predicted_class.description
        },
        confidence: backendResult.effective_score || backendResult.similarity_score || 0,
        similarityScore: backendResult.similarity_score || 0,
        rerankScore: backendResult.rerank_score,
        attributeScore: backendResult.attribute_score,
        attributeValidation: backendResult.attribute_validation ? {
          conditionsMet: backendResult.attribute_validation.conditions_met || [],
          conditionsNotMet: backendResult.attribute_validation.conditions_not_met || [],
          overallScore: backendResult.attribute_validation.overall_score || 0,
          details: backendResult.attribute_validation.evaluation_details || {},
          evaluationTree: backendResult.attribute_validation.evaluation_details?.evaluation_tree
        } : undefined,
        alternatives: (backendResult.alternatives || []).map((alt: any) => ({
          class: {
            id: alt.name.toLowerCase().replace(/\s+/g, '_'),
            name: alt.name,
            description: alt.description || 'Alternative classification'
          },
          confidence: alt.effective_score || alt.similarity_score || 0,
          scores: {
            similarity: alt.similarity_score || 0,
            rerank: alt.rerank_score,
            attribute: alt.attribute_score
          }
        })),
        processingTime: backendResult.processing_time,
        metadata: backendResult.metadata || {}
      };

      return {
        success: true,
        data: classificationResult,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);

      return {
        success: false,
        error: {
          type: 'processing',
          message: 'Classification failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check your input and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Classify PDF document using the specified dataset and configuration
   */
  async classifyPDF(request: ClassifyPDFRequest): Promise<ApiResponse<ClassificationResult>> {
    try {
      this.validatePDFRequest(request);

      const response = await apiClient.uploadFile<any>(
        '/classify/pdf',
        request.file,
        {
          dataset_id: request.datasetId,
          config_json: JSON.stringify({
            use_reranking: request.config.useReranking,
            reranking_model: request.config.rerankingModel || 'us.amazon.nova-lite-v1:0',
            use_attribute_validation: request.config.useAttributeValidation,
            top_k_candidates: request.config.topKCandidates,
          }),
        }
      );

      if (!response.success) {
        ErrorHandler.logError(response.error!);
        return response;
      }

      // Transform backend response to frontend format
      const backendResult = response.data;
      console.log('Backend PDF classification result:', backendResult); // Debug log

      const classificationResult: ClassificationResult = {
        predictedClass: {
          id: backendResult.predicted_class.name.toLowerCase().replace(/\s+/g, '_'),
          name: backendResult.predicted_class.name,
          description: backendResult.predicted_class.description
        },
        confidence: backendResult.effective_score || backendResult.similarity_score || 0,
        similarityScore: backendResult.similarity_score || 0,
        rerankScore: backendResult.rerank_score,
        attributeScore: backendResult.attribute_score,
        attributeValidation: backendResult.attribute_validation ? {
          conditionsMet: backendResult.attribute_validation.conditions_met || [],
          conditionsNotMet: backendResult.attribute_validation.conditions_not_met || [],
          overallScore: backendResult.attribute_validation.overall_score || 0,
          details: backendResult.attribute_validation.evaluation_details || {},
          evaluationTree: backendResult.attribute_validation.evaluation_details?.evaluation_tree
        } : undefined,
        alternatives: (backendResult.alternatives || []).map((alt: any) => ({
          class: {
            id: alt.name.toLowerCase().replace(/\s+/g, '_'),
            name: alt.name,
            description: alt.description || 'Alternative classification'
          },
          confidence: alt.effective_score || alt.similarity_score || 0,
          scores: {
            similarity: alt.similarity_score || 0,
            rerank: alt.rerank_score,
            attribute: alt.attribute_score
          }
        })),
        processingTime: backendResult.processing_time,
        metadata: {
          ...backendResult.metadata,
          pdf_metadata: backendResult.pdf_metadata
        }
      };

      return {
        success: true,
        data: classificationResult,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);

      return {
        success: false,
        error: {
          type: 'processing',
          message: 'PDF classification failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check your PDF file and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Extract content from PDF without classification
   */
  async extractPDFContent(file: File): Promise<ApiResponse<PDFExtractionResult>> {
    try {
      this.validatePDFFile(file);

      // For now, we'll use a mock extraction since the backend doesn't have a separate extract endpoint
      // In a real implementation, you might want to add this endpoint to the backend
      const mockResult: PDFExtractionResult = {
        extractedText: 'PDF content extraction would happen here...',
        pageCount: 1,
        extractionMethod: 'OCR',
        confidence: 0.95
      };

      return {
        success: true,
        data: mockResult,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);

      return {
        success: false,
        error: {
          type: 'processing',
          message: 'PDF extraction failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please check your PDF file and try again.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Validate text classification request
   */
  private validateTextRequest(request: ClassifyTextRequest): void {
    if (!request.text?.trim()) {
      throw ErrorHandler.createValidationError('Text content is required', 'text');
    }

    if (!request.datasetId?.trim()) {
      throw ErrorHandler.createValidationError('Dataset ID is required', 'datasetId');
    }

    if (!request.config) {
      throw ErrorHandler.createValidationError('Configuration is required', 'config');
    }

    if (request.config.topKCandidates < 1 || request.config.topKCandidates > 100) {
      throw ErrorHandler.createValidationError(
        'Top K candidates must be between 1 and 100',
        'config.topKCandidates'
      );
    }
  }

  /**
   * Validate PDF classification request
   */
  private validatePDFRequest(request: ClassifyPDFRequest): void {
    this.validatePDFFile(request.file);

    if (!request.datasetId?.trim()) {
      throw ErrorHandler.createValidationError('Dataset ID is required', 'datasetId');
    }

    if (!request.config) {
      throw ErrorHandler.createValidationError('Configuration is required', 'config');
    }
  }

  /**
   * Validate PDF file
   */
  private validatePDFFile(file: File): void {
    if (!file) {
      throw ErrorHandler.createValidationError('PDF file is required', 'file');
    }

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      throw ErrorHandler.createValidationError('Only PDF files are supported', 'file');
    }

    // Check file size (50MB limit)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      throw ErrorHandler.createValidationError(
        'PDF file size must be less than 50MB',
        'file'
      );
    }

    if (file.size === 0) {
      throw ErrorHandler.createValidationError('PDF file cannot be empty', 'file');
    }
  }
}

// Default classification service instance
export const classificationService = new ClassificationService();