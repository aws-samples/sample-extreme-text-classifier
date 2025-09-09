/**
 * Integration test for embedding generation functionality
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import { DatasetList } from '../DatasetList';
import { useDatasetStore } from '../../../stores/dataset-store';

// Mock the API client
jest.mock('../../../services/api-client', () => ({
  apiClient: {
    post: jest.fn(),
    get: jest.fn(),
  }
}));

describe.skip('Embedding Generation Integration', () => {
  const mockOnSelectDataset = jest.fn();
  const mockOnCreateNew = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the store
    useDatasetStore.getState().reset();
  });

  test('embedding generation status updates correctly', () => {
    const store = useDatasetStore.getState();
    
    // Test setting embedding generation status
    store.setEmbeddingGenerationStatus('test-dataset', {
      isGenerating: true,
      progress: 50,
      message: 'Processing classes...'
    });

    expect(store.embeddingGenerationStatus['test-dataset']).toEqual({
      isGenerating: true,
      progress: 50,
      message: 'Processing classes...'
    });

    // Test updating progress
    store.setEmbeddingGenerationStatus('test-dataset', {
      isGenerating: true,
      progress: 100,
      message: 'Complete!'
    });

    expect(store.embeddingGenerationStatus['test-dataset'].progress).toBe(100);
    expect(store.embeddingGenerationStatus['test-dataset'].message).toBe('Complete!');
  });

  test('datasets display embedding status correctly', () => {
    // Mock datasets that need embeddings
    const datasetsNeedingEmbeddings = [
      {
        id: 'dataset-1',
        name: 'Dataset 1',
        description: 'First dataset',
        classes: [],
        embeddingsGenerated: false,
        createdAt: new Date(),
        updatedAt: new Date()
      },
      {
        id: 'dataset-2', 
        name: 'Dataset 2',
        description: 'Second dataset',
        classes: [],
        embeddingsGenerated: false,
        createdAt: new Date(),
        updatedAt: new Date()
      }
    ];

    // Add datasets to store
    const store = useDatasetStore.getState();
    datasetsNeedingEmbeddings.forEach(dataset => {
      store.addDataset(dataset);
    });

    render(
      <DatasetList
        onSelectDataset={mockOnSelectDataset}
        onCreateNew={mockOnCreateNew}
      />
    );

    // Verify both datasets show pending embeddings
    expect(screen.getAllByText('Embeddings Pending')).toHaveLength(2);
    expect(screen.getAllByText('Generate')).toHaveLength(2);
  });

  test('dataset with embeddings shows ready status', () => {
    const datasetWithEmbeddings = {
      id: 'dataset-ready',
      name: 'Ready Dataset',
      description: 'Dataset with embeddings',
      classes: [
        {
          id: 'class-1',
          name: 'Test Class',
          description: 'A test class'
        }
      ],
      embeddingsGenerated: true,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    // Add dataset to store
    const store = useDatasetStore.getState();
    store.addDataset(datasetWithEmbeddings);

    render(
      <DatasetList
        onSelectDataset={mockOnSelectDataset}
        onCreateNew={mockOnCreateNew}
      />
    );

    // Verify dataset shows ready status
    expect(screen.getByText('Ready Dataset')).toBeInTheDocument();
    expect(screen.getByText('Embeddings Ready')).toBeInTheDocument();
  });
});