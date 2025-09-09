/**
 * Store tests - Basic tests for Zustand stores
 */

import { renderHook, act } from '@testing-library/react';
import { useDatasetStore } from '../dataset-store';
import { useClassificationStore } from '../classification-store';
import { useUIStore } from '../ui-store';

describe('Dataset Store', () => {
  beforeEach(() => {
    useDatasetStore.getState().reset();
  });

  test('should initialize with empty state', () => {
    const { result } = renderHook(() => useDatasetStore());
    
    expect(result.current.datasets).toEqual([]);
    expect(result.current.selectedDataset).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  test('should add dataset', () => {
    const { result } = renderHook(() => useDatasetStore());
    
    const mockDataset = {
      id: 'test-1',
      name: 'Test Dataset',
      description: 'A test dataset',
      classes: [],
      embeddingsGenerated: false,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    act(() => {
      result.current.addDataset(mockDataset);
    });

    expect(result.current.datasets).toHaveLength(1);
    expect(result.current.datasets[0]).toEqual(mockDataset);
  });

  test('should set selected dataset', () => {
    const { result } = renderHook(() => useDatasetStore());
    
    const mockDataset = {
      id: 'test-1',
      name: 'Test Dataset',
      description: 'A test dataset',
      classes: [],
      embeddingsGenerated: false,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    act(() => {
      result.current.setSelectedDataset(mockDataset);
    });

    expect(result.current.selectedDataset).toEqual(mockDataset);
  });

  test('should set embedding generation status', () => {
    const { result } = renderHook(() => useDatasetStore());
    
    const status = {
      isGenerating: true,
      progress: 50,
      message: 'Generating embeddings...'
    };

    act(() => {
      result.current.setEmbeddingGenerationStatus('test-dataset', status);
    });

    expect(result.current.embeddingGenerationStatus['test-dataset']).toEqual(status);
  });

  test('should initialize with empty embedding generation status', () => {
    const { result } = renderHook(() => useDatasetStore());
    
    expect(result.current.embeddingGenerationStatus).toEqual({});
  });
});

describe('Classification Store', () => {
  beforeEach(() => {
    useClassificationStore.getState().reset();
  });

  test('should initialize with default config', () => {
    const { result } = renderHook(() => useClassificationStore());
    
    expect(result.current.config).toEqual({
      useReranking: false,
      useAttributeValidation: false,
      topKCandidates: 5
    });
    expect(result.current.currentResult).toBeNull();
    expect(result.current.isClassifying).toBe(false);
  });

  test('should update config', () => {
    const { result } = renderHook(() => useClassificationStore());
    
    act(() => {
      result.current.updateConfig({ useReranking: true, topKCandidates: 10 });
    });

    expect(result.current.config.useReranking).toBe(true);
    expect(result.current.config.topKCandidates).toBe(10);
    expect(result.current.config.useAttributeValidation).toBe(false); // Should remain unchanged
  });

  test('should add result to history', () => {
    const { result } = renderHook(() => useClassificationStore());
    
    const mockResult = {
      predictedClass: {
        id: 'class-1',
        name: 'Test Class',
        description: 'A test class'
      },
      confidence: 0.85,
      similarityScore: 0.82,
      alternatives: []
    };

    act(() => {
      result.current.addToHistory(mockResult);
    });

    expect(result.current.resultHistory).toHaveLength(1);
    expect(result.current.resultHistory[0]).toEqual(mockResult);
  });
});

describe('UI Store', () => {
  beforeEach(() => {
    useUIStore.getState().reset();
  });

  test('should initialize with default state', () => {
    const { result } = renderHook(() => useUIStore());
    
    expect(result.current.loading.isLoading).toBe(false);
    expect(result.current.toasts).toEqual([]);
    expect(result.current.modal.isOpen).toBe(false);
    expect(result.current.currentSection).toBe('dashboard');
  });

  test('should set loading state', () => {
    const { result } = renderHook(() => useUIStore());
    
    act(() => {
      result.current.setLoading({ isLoading: true, operation: 'test' });
    });

    expect(result.current.loading.isLoading).toBe(true);
    expect(result.current.loading.operation).toBe('test');
  });

  test('should add toast', () => {
    const { result } = renderHook(() => useUIStore());
    
    act(() => {
      result.current.addToast({
        type: 'success',
        title: 'Test Toast',
        message: 'This is a test'
      });
    });

    expect(result.current.toasts).toHaveLength(1);
    expect(result.current.toasts[0].title).toBe('Test Toast');
    expect(result.current.toasts[0].type).toBe('success');
  });

  test('should open and close modal', () => {
    const { result } = renderHook(() => useUIStore());
    
    act(() => {
      result.current.openModal({
        type: 'confirm',
        title: 'Test Modal',
        content: 'Are you sure?'
      });
    });

    expect(result.current.modal.isOpen).toBe(true);
    expect(result.current.modal.title).toBe('Test Modal');

    act(() => {
      result.current.closeModal();
    });

    expect(result.current.modal.isOpen).toBe(false);
  });

  test('should toggle sidebar', () => {
    const { result } = renderHook(() => useUIStore());
    
    expect(result.current.preferences.sidebarCollapsed).toBe(false);

    act(() => {
      result.current.toggleSidebar();
    });

    expect(result.current.preferences.sidebarCollapsed).toBe(true);
  });
});