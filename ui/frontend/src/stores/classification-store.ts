/**
 * Classification Store - Manages classification results and configurations using Zustand
 */

import { create } from 'zustand';
import { 
  ClassificationResult, 
  ClassificationConfig, 
  ClassifyTextRequest,
  ClassifyPDFRequest,
  PDFExtractionResult
} from '../types/classification';
import { AppError } from '../types/errors';

interface ClassificationState {
  // State
  currentResult: ClassificationResult | null;
  resultHistory: ClassificationResult[];
  config: ClassificationConfig;
  isClassifying: boolean;
  error: AppError | null;
  pdfExtractionResult: PDFExtractionResult | null;

  // Actions
  setCurrentResult: (result: ClassificationResult | null) => void;
  addToHistory: (result: ClassificationResult) => void;
  clearHistory: () => void;
  setConfig: (config: ClassificationConfig) => void;
  updateConfig: (updates: Partial<ClassificationConfig>) => void;
  setClassifying: (classifying: boolean) => void;
  setError: (error: AppError | null) => void;
  setPdfExtractionResult: (result: PDFExtractionResult | null) => void;

  // Classification Operations
  classifyText: (request: ClassifyTextRequest) => Promise<ClassificationResult>;
  classifyPDF: (request: ClassifyPDFRequest) => Promise<ClassificationResult>;
  
  // Utility actions
  clearError: () => void;
  clearPdfExtractionResult: () => void;
  reset: () => void;
}

// Default classification configuration
const defaultConfig: ClassificationConfig = {
  useReranking: false,
  rerankingModel: 'us.amazon.nova-lite-v1:0',
  useAttributeValidation: false,
  topKCandidates: 5
};

export const useClassificationStore = create<ClassificationState>((set, get) => ({
  // Initial state
  currentResult: null,
  resultHistory: [],
  config: defaultConfig,
  isClassifying: false,
  error: null,
  pdfExtractionResult: null,

  // Basic setters
  setCurrentResult: (result) => set({ currentResult: result }),
  addToHistory: (result) => set((state) => ({
    resultHistory: [result, ...state.resultHistory].slice(0, 50) // Keep last 50 results
  })),
  clearHistory: () => set({ resultHistory: [] }),
  setConfig: (config) => set({ config }),
  updateConfig: (updates) => set((state) => ({
    config: { ...state.config, ...updates }
  })),
  setClassifying: (classifying) => set({ isClassifying: classifying }),
  setError: (error) => set({ error }),
  setPdfExtractionResult: (result) => set({ pdfExtractionResult: result }),

  // Classification Operations
  classifyText: async (request) => {
    set({ isClassifying: true, error: null });
    try {
      // Import the classification service dynamically to avoid circular dependencies
      const { classificationService } = await import('../services/classification-service');
      
      const response = await classificationService.classifyText(request);
      
      if (!response.success) {
        throw new Error(response.error?.message || 'Classification failed');
      }
      
      const result = response.data!;
      set({ currentResult: result, isClassifying: false });
      get().addToHistory(result);
      return result;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'CLASSIFICATION_FAILED',
        message: 'Failed to classify text',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isClassifying: false });
      throw appError;
    }
  },

  classifyPDF: async (request) => {
    set({ isClassifying: true, error: null });
    try {
      // Import the classification service dynamically to avoid circular dependencies
      const { classificationService } = await import('../services/classification-service');
      
      const response = await classificationService.classifyPDF(request);
      
      if (!response.success) {
        throw new Error(response.error?.message || 'PDF classification failed');
      }
      
      const result = response.data!;
      
      // Extract PDF metadata if available
      if (result.metadata?.pdf_metadata) {
        const pdfMeta = result.metadata.pdf_metadata;
        const extractionResult: PDFExtractionResult = {
          extractedText: pdfMeta.extracted_text_preview || 'Text extracted from PDF',
          pageCount: pdfMeta.page_count || 1,
          extractionMethod: 'Multimodal LLM',
          confidence: 0.95
        };
        set({ pdfExtractionResult: extractionResult });
      }
      
      set({ currentResult: result, isClassifying: false });
      get().addToHistory(result);
      return result;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'PDF_CLASSIFICATION_FAILED',
        message: 'Failed to classify PDF',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isClassifying: false });
      throw appError;
    }
  },



  // Utility actions
  clearError: () => set({ error: null }),
  clearPdfExtractionResult: () => set({ pdfExtractionResult: null }),
  reset: () => set({
    currentResult: null,
    resultHistory: [],
    config: defaultConfig,
    isClassifying: false,
    error: null,
    pdfExtractionResult: null
  })
}));