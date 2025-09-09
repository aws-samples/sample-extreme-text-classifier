/**
 * Attribute Store - Manages class attributes and validation rules using Zustand
 */

import { create } from 'zustand';
import { ClassAttribute, GenerateAttributesRequest, SaveAttributesRequest } from '../types/attributes';
import { AppError } from '../types/errors';

interface AttributeState {
  // State
  attributes: Record<string, ClassAttribute[]>; // datasetId -> attributes
  selectedAttribute: ClassAttribute | null;
  isLoading: boolean;
  isGenerating: boolean;
  error: AppError | null;
  generationProgress: {
    datasetId: string;
    progress: number;
    message: string;
  } | null;

  // Actions
  setAttributes: (datasetId: string, attributes: ClassAttribute[]) => void;
  addAttribute: (datasetId: string, attribute: ClassAttribute) => void;
  updateAttribute: (datasetId: string, attribute: ClassAttribute) => void;
  removeAttribute: (datasetId: string, classId: string) => void;
  setSelectedAttribute: (attribute: ClassAttribute | null) => void;
  setLoading: (loading: boolean) => void;
  setGenerating: (generating: boolean) => void;
  setError: (error: AppError | null) => void;
  setGenerationProgress: (progress: { datasetId: string; progress: number; message: string } | null) => void;

  // API Operations
  loadAttributes: (datasetId: string) => Promise<ClassAttribute[]>;
  generateAttributes: (request: GenerateAttributesRequest) => Promise<ClassAttribute[]>;
  saveAttributes: (request: SaveAttributesRequest) => Promise<void>;
  deleteAttributes: (datasetId: string) => Promise<void>;

  // Utility actions
  getAttributesForDataset: (datasetId: string) => ClassAttribute[];
  hasAttributes: (datasetId: string) => boolean;
  clearError: () => void;
  reset: () => void;
}

export const useAttributeStore = create<AttributeState>((set, get) => ({
  // Initial state
  attributes: {},
  selectedAttribute: null,
  isLoading: false,
  isGenerating: false,
  error: null,
  generationProgress: null,

  // Basic setters
  setAttributes: (datasetId, attributes) => set((state) => ({
    attributes: { ...state.attributes, [datasetId]: attributes }
  })),
  addAttribute: (datasetId, attribute) => set((state) => ({
    attributes: {
      ...state.attributes,
      [datasetId]: [...(state.attributes[datasetId] || []), attribute]
    }
  })),
  updateAttribute: (datasetId, attribute) => set((state) => ({
    attributes: {
      ...state.attributes,
      [datasetId]: (state.attributes[datasetId] || []).map(attr =>
        attr.classId === attribute.classId ? attribute : attr
      )
    },
    selectedAttribute: state.selectedAttribute?.classId === attribute.classId ? attribute : state.selectedAttribute
  })),
  removeAttribute: (datasetId, classId) => set((state) => ({
    attributes: {
      ...state.attributes,
      [datasetId]: (state.attributes[datasetId] || []).filter(attr => attr.classId !== classId)
    },
    selectedAttribute: state.selectedAttribute?.classId === classId ? null : state.selectedAttribute
  })),
  setSelectedAttribute: (attribute) => set({ selectedAttribute: attribute }),
  setLoading: (loading) => set({ isLoading: loading }),
  setGenerating: (generating) => set({ isGenerating: generating }),
  setError: (error) => set({ error }),
  setGenerationProgress: (progress) => set({ generationProgress: progress }),

  // API Operations
  loadAttributes: async (datasetId) => {
    set({ isLoading: true, error: null });
    try {
      // Import the attribute service dynamically to avoid circular dependencies
      const { attributeService } = await import('../services/attribute-service');

      const response = await attributeService.getAttributes(datasetId);

      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to load attributes');
      }

      const attributes = response.data || [];
      get().setAttributes(datasetId, attributes);
      set({ isLoading: false });
      return attributes;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'ATTRIBUTES_LOAD_FAILED',
        message: 'Failed to load attributes',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  generateAttributes: async (request) => {
    set({ isGenerating: true, error: null });

    try {
      // Import the attribute service dynamically to avoid circular dependencies
      const { attributeService } = await import('../services/attribute-service');

      // Set initial progress and start API call immediately
      get().setGenerationProgress({
        datasetId: request.datasetId,
        progress: 10,
        message: 'Starting attribute generation with AI models... This may take 3-5 minutes.'
      });

      // Start gradual progress animation during API call
      const progressSteps = [
        { progress: 20, message: 'Analyzing class definitions with AI...', delay: 5000 },
        { progress: 35, message: 'Generating validation rules...', delay: 10000 },
        { progress: 50, message: 'Processing class relationships...', delay: 15000 },
        { progress: 65, message: 'Creating logical conditions...', delay: 20000 },
        { progress: 80, message: 'Optimizing attribute rules...', delay: 25000 },
        { progress: 90, message: 'Finalizing attributes...', delay: 30000 }
      ];

      // Track timeout IDs so we can cancel them if API completes early
      const timeoutIds: NodeJS.Timeout[] = [];
      let isCompleted = false;

      // Start the progress animation
      progressSteps.forEach(step => {
        const timeoutId = setTimeout(() => {
          // Only update progress if generation hasn't completed yet
          if (!isCompleted) {
            get().setGenerationProgress({
              datasetId: request.datasetId,
              progress: step.progress,
              message: step.message
            });
          }
        }, step.delay);
        timeoutIds.push(timeoutId);
      });

      // Call the actual API (this will take 1-2 minutes)
      const response = await attributeService.generateAttributes(request);
      
      // Mark as completed and cancel any remaining timeouts
      isCompleted = true;
      timeoutIds.forEach(id => clearTimeout(id));

      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to generate attributes');
      }

      // Show completion
      get().setGenerationProgress({
        datasetId: request.datasetId,
        progress: 100,
        message: 'Attributes generated successfully!'
      });

      const attributes = response.data || [];
      console.log('Generated attributes received:', attributes);
      get().setAttributes(request.datasetId, attributes);

      // Brief delay to show completion before clearing
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Clear generation progress
      set({ isGenerating: false, generationProgress: null });

      console.log('Attribute generation completed, progress cleared');
      return attributes;
    } catch (error) {
      console.error('Attribute generation failed:', error);
      const appError: AppError = {
        type: 'processing',
        code: 'ATTRIBUTES_GENERATION_FAILED',
        message: 'Failed to generate attributes',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isGenerating: false, generationProgress: null });
      throw appError;
    }
  },

  saveAttributes: async (request) => {
    set({ isLoading: true, error: null });
    try {
      // Import the attribute service dynamically to avoid circular dependencies
      const { attributeService } = await import('../services/attribute-service');

      const response = await attributeService.saveAttributes(request);

      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to save attributes');
      }

      // Update local state
      get().setAttributes(request.datasetId, request.attributes);
      set({ isLoading: false });
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'ATTRIBUTES_SAVE_FAILED',
        message: 'Failed to save attributes',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  deleteAttributes: async (datasetId) => {
    set({ isLoading: true, error: null });
    try {
      // Import the attribute service dynamically to avoid circular dependencies
      const { attributeService } = await import('../services/attribute-service');

      const response = await attributeService.deleteAttributes(datasetId);

      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to delete attributes');
      }

      // Remove from local state
      set((state) => {
        const newAttributes = { ...state.attributes };
        delete newAttributes[datasetId];
        return { attributes: newAttributes, isLoading: false };
      });
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'ATTRIBUTES_DELETE_FAILED',
        message: 'Failed to delete attributes',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  // Utility actions
  getAttributesForDataset: (datasetId) => {
    return get().attributes[datasetId] || [];
  },

  hasAttributes: (datasetId) => {
    const attributes = get().attributes[datasetId];
    return attributes && attributes.length > 0;
  },

  clearError: () => set({ error: null }),

  reset: () => set({
    attributes: {},
    selectedAttribute: null,
    isLoading: false,
    isGenerating: false,
    error: null,
    generationProgress: null
  })
}));