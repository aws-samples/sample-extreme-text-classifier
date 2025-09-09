/**
 * Dataset Store - Manages loaded datasets and classes using Zustand
 */

import { create } from 'zustand';
import { Dataset, DatasetMetadata, ClassDefinition, CreateDatasetRequest, UpdateDatasetRequest } from '../types/dataset';
import { AppError } from '../types/errors';

interface DatasetState {
  // State
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  datasetMetadata: DatasetMetadata[];
  isLoading: boolean;
  error: AppError | null;
  embeddingGenerationStatus: Record<string, {
    isGenerating: boolean;
    progress: number;
    message: string;
  }>;

  // Actions
  setDatasets: (datasets: Dataset[]) => void;
  addDataset: (dataset: Dataset) => void;
  updateDataset: (dataset: Dataset) => void;
  removeDataset: (datasetId: string) => void;
  setSelectedDataset: (dataset: Dataset | null) => void;
  setDatasetMetadata: (metadata: DatasetMetadata[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: AppError | null) => void;
  setEmbeddingGenerationStatus: (datasetId: string, status: {
    isGenerating: boolean;
    progress: number;
    message: string;
  }) => void;
  
  // CRUD Operations
  createDataset: (request: CreateDatasetRequest) => Promise<Dataset>;
  loadDataset: (datasetId: string) => Promise<Dataset>;
  loadAllDatasets: () => Promise<Dataset[]>;
  updateDatasetData: (request: UpdateDatasetRequest) => Promise<Dataset>;
  deleteDataset: (datasetId: string) => Promise<void>;
  
  // Embedding Operations
  generateEmbeddings: (datasetId: string) => Promise<void>;
  
  // Class management
  addClass: (datasetId: string, classDefinition: Omit<ClassDefinition, 'id'>) => Promise<ClassDefinition>;
  updateClass: (datasetId: string, classDefinition: ClassDefinition) => Promise<ClassDefinition>;
  removeClass: (datasetId: string, classId: string) => Promise<void>;
  
  // Utility actions
  clearError: () => void;
  reset: () => void;
}

// No sample datasets - they will be loaded from the backend

export const useDatasetStore = create<DatasetState>((set, get) => ({
  // Initial state - empty, will be loaded from backend
  datasets: [],
  selectedDataset: null,
  datasetMetadata: [],
  isLoading: false,
  error: null,
  embeddingGenerationStatus: {},

  // Basic setters
  setDatasets: (datasets) => set({ datasets }),
  addDataset: (dataset) => set((state) => ({ 
    datasets: [...state.datasets, dataset] 
  })),
  updateDataset: (dataset) => set((state) => ({
    datasets: state.datasets.map(d => d.id === dataset.id ? dataset : d),
    selectedDataset: state.selectedDataset?.id === dataset.id ? dataset : state.selectedDataset
  })),
  removeDataset: (datasetId) => set((state) => ({
    datasets: state.datasets.filter(d => d.id !== datasetId),
    selectedDataset: state.selectedDataset?.id === datasetId ? null : state.selectedDataset
  })),
  setSelectedDataset: (dataset) => set({ selectedDataset: dataset }),
  setDatasetMetadata: (metadata) => set({ datasetMetadata: metadata }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setEmbeddingGenerationStatus: (datasetId, status) => set((state) => ({
    embeddingGenerationStatus: {
      ...state.embeddingGenerationStatus,
      [datasetId]: status
    }
  })),

  // CRUD Operations
  createDataset: async (request) => {
    set({ isLoading: true, error: null });
    try {
      // Import the dataset service dynamically to avoid circular dependencies
      const { datasetService } = await import('../services/dataset-service');
      
      const response = await datasetService.saveDataset(request);
      
      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to create dataset');
      }
      
      const newDataset = response.data!;
      get().addDataset(newDataset);
      set({ isLoading: false });
      return newDataset;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'DATASET_CREATE_FAILED',
        message: 'Failed to create dataset',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  loadDataset: async (datasetId) => {
    set({ isLoading: true, error: null });
    try {
      // Import the dataset service dynamically to avoid circular dependencies
      const { datasetService } = await import('../services/dataset-service');
      
      const response = await datasetService.loadDataset(datasetId);
      
      if (!response.success) {
        throw new Error(response.error?.message || 'Dataset not found');
      }
      
      const dataset = response.data!;
      set({ selectedDataset: dataset, isLoading: false });
      return dataset;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'DATASET_LOAD_FAILED',
        message: 'Failed to load dataset',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  loadAllDatasets: async () => {
    set({ isLoading: true, error: null });
    try {
      // Import the dataset service dynamically to avoid circular dependencies
      const { datasetService } = await import('../services/dataset-service');
      
      const response = await datasetService.listDatasets();
      
      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to load datasets');
      }
      
      // Convert metadata to full datasets by loading each one
      const metadata = response.data!;
      const loadedDatasets: Dataset[] = [];
      
      for (const meta of metadata) {
        try {
          const datasetResponse = await datasetService.loadDataset(meta.id);
          if (datasetResponse.success) {
            loadedDatasets.push(datasetResponse.data!);
          }
        } catch (error) {
          console.warn(`Failed to load dataset ${meta.id}:`, error);
        }
      }
      
      set({ datasets: loadedDatasets, isLoading: false });
      
      // Note: Removed automatic embedding generation to avoid unwanted regeneration
      // Users can manually generate embeddings using the "Generate Embeddings" button
      
      return loadedDatasets;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'DATASETS_LOAD_FAILED',
        message: 'Failed to load datasets',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  updateDatasetData: async (request) => {
    set({ isLoading: true, error: null });
    try {
      // Import the dataset service dynamically to avoid circular dependencies
      const { datasetService } = await import('../services/dataset-service');
      
      const response = await datasetService.saveDataset(request);
      
      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to update dataset');
      }
      
      const updatedDataset = response.data!;
      get().updateDataset(updatedDataset);
      set({ isLoading: false });
      return updatedDataset;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'DATASET_UPDATE_FAILED',
        message: 'Failed to update dataset',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  deleteDataset: async (datasetId) => {
    set({ isLoading: true, error: null });
    try {
      // Import the dataset service dynamically to avoid circular dependencies
      const { datasetService } = await import('../services/dataset-service');
      
      const response = await datasetService.deleteDataset(datasetId);
      
      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to delete dataset');
      }
      
      get().removeDataset(datasetId);
      set({ isLoading: false });
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'DATASET_DELETE_FAILED',
        message: 'Failed to delete dataset',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  // Class management
  addClass: async (datasetId, classDefinition) => {
    set({ isLoading: true, error: null });
    try {
      const dataset = get().datasets.find(d => d.id === datasetId);
      if (!dataset) {
        throw new Error('Dataset not found');
      }

      const newClass: ClassDefinition = {
        ...classDefinition,
        id: `class_${Date.now()}`
      };

      const updatedDataset: Dataset = {
        ...dataset,
        classes: [...dataset.classes, newClass],
        updatedAt: new Date()
      };

      get().updateDataset(updatedDataset);
      set({ isLoading: false });
      return newClass;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'CLASS_ADD_FAILED',
        message: 'Failed to add class',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  updateClass: async (datasetId, classDefinition) => {
    set({ isLoading: true, error: null });
    try {
      const dataset = get().datasets.find(d => d.id === datasetId);
      if (!dataset) {
        throw new Error('Dataset not found');
      }

      const updatedDataset: Dataset = {
        ...dataset,
        classes: dataset.classes.map(c => 
          c.id === classDefinition.id ? classDefinition : c
        ),
        updatedAt: new Date()
      };

      get().updateDataset(updatedDataset);
      set({ isLoading: false });
      return classDefinition;
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'CLASS_UPDATE_FAILED',
        message: 'Failed to update class',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  removeClass: async (datasetId, classId) => {
    set({ isLoading: true, error: null });
    try {
      const dataset = get().datasets.find(d => d.id === datasetId);
      if (!dataset) {
        throw new Error('Dataset not found');
      }

      const updatedDataset: Dataset = {
        ...dataset,
        classes: dataset.classes.filter(c => c.id !== classId),
        updatedAt: new Date()
      };

      get().updateDataset(updatedDataset);
      set({ isLoading: false });
    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'CLASS_REMOVE_FAILED',
        message: 'Failed to remove class',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      set({ error: appError, isLoading: false });
      throw appError;
    }
  },

  // Embedding Operations
  generateEmbeddings: async (datasetId) => {
    const { setEmbeddingGenerationStatus, updateDataset } = get();
    
    try {
      // Set initial status
      setEmbeddingGenerationStatus(datasetId, {
        isGenerating: true,
        progress: 0,
        message: 'Initializing embedding generation...'
      });

      // Import the dataset service dynamically to avoid circular dependencies
      const { datasetService } = await import('../services/dataset-service');
      
      // Simulate progress updates
      const progressSteps = [
        { progress: 20, message: 'Loading dataset classes...' },
        { progress: 40, message: 'Generating embeddings for class definitions...' },
        { progress: 60, message: 'Processing example texts...' },
        { progress: 80, message: 'Building vector index...' },
        { progress: 100, message: 'Embeddings generated successfully!' }
      ];

      for (const step of progressSteps) {
        setEmbeddingGenerationStatus(datasetId, {
          isGenerating: true,
          progress: step.progress,
          message: step.message
        });
        
        // Add delay to show progress
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Call the actual API
      const response = await datasetService.generateEmbeddings(datasetId);
      
      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to generate embeddings');
      }

      // Update dataset status
      const dataset = get().datasets.find(d => d.id === datasetId);
      if (dataset) {
        const updatedDataset = {
          ...dataset,
          embeddingsGenerated: true,
          updatedAt: new Date()
        };
        updateDataset(updatedDataset);
      }

      // Clear generation status
      setEmbeddingGenerationStatus(datasetId, {
        isGenerating: false,
        progress: 100,
        message: 'Embeddings ready!'
      });

      // Clear status after a delay
      setTimeout(() => {
        set((state) => {
          const newStatus = { ...state.embeddingGenerationStatus };
          delete newStatus[datasetId];
          return { embeddingGenerationStatus: newStatus };
        });
      }, 3000);

    } catch (error) {
      const appError: AppError = {
        type: 'processing',
        code: 'EMBEDDING_GENERATION_FAILED',
        message: 'Failed to generate embeddings',
        details: error instanceof Error ? error.message : 'Unknown error',
        recoverable: true,
        timestamp: new Date()
      };
      
      setEmbeddingGenerationStatus(datasetId, {
        isGenerating: false,
        progress: 0,
        message: 'Failed to generate embeddings'
      });
      
      set({ error: appError });
      throw appError;
    }
  },

  // Utility actions
  clearError: () => set({ error: null }),
  reset: () => set({
    datasets: [],
    selectedDataset: null,
    datasetMetadata: [],
    isLoading: false,
    error: null,
    embeddingGenerationStatus: {}
  })
}));