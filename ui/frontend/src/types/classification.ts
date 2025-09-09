/**
 * Classification-related TypeScript interfaces and types
 */

import { ClassDefinition } from './dataset';
import { AttributeValidationResult } from './attributes';

export interface ClassificationConfig {
  useReranking: boolean;
  rerankingModel?: string;
  rerankingParams?: Record<string, any>;
  useAttributeValidation: boolean;
  topKCandidates: number;
}

export interface ClassificationScores {
  similarity: number;
  rerank?: number;
  attribute?: number;
}

export interface ClassificationAlternative {
  class: ClassDefinition;
  confidence: number;
  scores: ClassificationScores;
}

export interface ClassificationResult {
  predictedClass: ClassDefinition;
  confidence: number;
  similarityScore: number;
  rerankScore?: number;
  attributeScore?: number;
  alternatives: ClassificationAlternative[];
  attributeValidation?: AttributeValidationResult;
  processingTime?: number;
  metadata?: Record<string, any>;
}



export interface ClassifyTextRequest {
  text: string;
  datasetId: string;
  config: ClassificationConfig;
}

export interface ClassifyPDFRequest {
  file: File;
  datasetId: string;
  config: ClassificationConfig;
}

export interface PDFExtractionResult {
  extractedText: string;
  pageCount: number;
  extractionMethod: string;
  confidence: number;
}