/**
 * API-related TypeScript interfaces and types
 */

import { Dataset, DatasetMetadata, CreateDatasetRequest, UpdateDatasetRequest } from './dataset';
import { ClassAttribute, GenerateAttributesRequest, SaveAttributesRequest } from './attributes';
import { ClassificationResult, ClassifyTextRequest, ClassifyPDFRequest, PDFExtractionResult } from './classification';

// Base API response wrapper
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiError;
  message?: string;
  timestamp: string;
}

// API Error structure
export interface ApiError {
  type: 'network' | 'validation' | 'processing' | 'filesystem' | 'auth';
  message: string;
  details?: string;
  code?: string;
  recoverable: boolean;
  suggestedAction?: string;
}

// Dataset API interfaces
export interface DatasetAPI {
  listDatasets(): Promise<ApiResponse<DatasetMetadata[]>>;
  loadDataset(id: string): Promise<ApiResponse<Dataset>>;
  saveDataset(dataset: CreateDatasetRequest | UpdateDatasetRequest): Promise<ApiResponse<Dataset>>;
  deleteDataset(id: string): Promise<ApiResponse<void>>;
  generateDataset(domain: string, classCount: number): Promise<ApiResponse<Dataset>>;
  generateEmbeddings(datasetId: string): Promise<ApiResponse<void>>;
}

// Classification API interfaces
export interface ClassificationAPI {
  classifyText(request: ClassifyTextRequest): Promise<ApiResponse<ClassificationResult>>;
  classifyPDF(request: ClassifyPDFRequest): Promise<ApiResponse<ClassificationResult>>;
  extractPDFContent(file: File): Promise<ApiResponse<PDFExtractionResult>>;
}

// Attribute API interfaces
export interface AttributeAPI {
  getAttributes(datasetId: string): Promise<ApiResponse<ClassAttribute[]>>;
  generateAttributes(request: GenerateAttributesRequest): Promise<ApiResponse<ClassAttribute[]>>;
  saveAttributes(request: SaveAttributesRequest): Promise<ApiResponse<void>>;
}

// Wizard API interfaces
export interface WizardAPI {
  generateDomainDataset(domain: string, options?: {
    classCount?: number;
    includeAttributes?: boolean;
    includeExamples?: boolean;
  }): Promise<ApiResponse<{
    dataset: Dataset;
    attributes?: ClassAttribute[];
    examples?: string[];
  }>>;
}

// File upload progress
export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

// API client configuration
export interface ApiClientConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
}

// HTTP methods
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

// Request options
export interface RequestOptions {
  method: HttpMethod;
  headers?: Record<string, string>;
  body?: any;
  timeout?: number;
  onUploadProgress?: (progress: UploadProgress) => void;
}