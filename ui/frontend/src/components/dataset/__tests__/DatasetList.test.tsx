/**
 * DatasetList component tests
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DatasetList } from '../DatasetList';
import { useDatasetStore } from '../../../stores/dataset-store';

// Mock the dataset store
jest.mock('../../../stores/dataset-store');

const mockUseDatasetStore = useDatasetStore as jest.MockedFunction<typeof useDatasetStore>;

describe('DatasetList', () => {
  const mockOnSelectDataset = jest.fn();
  const mockOnCreateNew = jest.fn();
  const mockGenerateEmbeddings = jest.fn();
  const mockDeleteDataset = jest.fn();
  const mockLoadAllDatasets = jest.fn();
  const mockClearError = jest.fn();

  const mockDataset = {
    id: 'test-dataset',
    name: 'Test Dataset',
    description: 'A test dataset for classification',
    classes: [
      {
        id: 'class-1',
        name: 'Class 1',
        description: 'First test class',
        examples: ['Example 1', 'Example 2']
      }
    ],
    embeddingsGenerated: false,
    createdAt: new Date('2024-01-01'),
    updatedAt: new Date('2024-01-02')
  };

  beforeEach(() => {
    jest.clearAllMocks();
    
    mockUseDatasetStore.mockReturnValue({
      datasets: [mockDataset],
      isLoading: false,
      error: null,
      embeddingGenerationStatus: {},
      loadAllDatasets: mockLoadAllDatasets,
      deleteDataset: mockDeleteDataset,
      generateEmbeddings: mockGenerateEmbeddings,
      clearError: mockClearError,
      // Add other required store properties with default values
      selectedDataset: null,
      datasetMetadata: [],
      setDatasets: jest.fn(),
      addDataset: jest.fn(),
      updateDataset: jest.fn(),
      removeDataset: jest.fn(),
      setSelectedDataset: jest.fn(),
      setDatasetMetadata: jest.fn(),
      setLoading: jest.fn(),
      setError: jest.fn(),
      setEmbeddingGenerationStatus: jest.fn(),
      createDataset: jest.fn(),
      loadDataset: jest.fn(),
      updateDatasetData: jest.fn(),
      addClass: jest.fn(),
      updateClass: jest.fn(),
      removeClass: jest.fn(),
      reset: jest.fn()
    });
  });

  test('renders dataset list with embedding generation button', () => {
    render(
      <DatasetList
        onSelectDataset={mockOnSelectDataset}
        onCreateNew={mockOnCreateNew}
      />
    );

    expect(screen.getByText('Test Dataset')).toBeInTheDocument();
    expect(screen.getByText('Embeddings Pending')).toBeInTheDocument();
    expect(screen.getByText('Generate')).toBeInTheDocument();
  });

  test('shows embedding generation progress', () => {
    mockUseDatasetStore.mockReturnValue({
      datasets: [mockDataset],
      isLoading: false,
      error: null,
      embeddingGenerationStatus: {
        'test-dataset': {
          isGenerating: true,
          progress: 75,
          message: 'Processing example texts...'
        }
      },
      loadAllDatasets: mockLoadAllDatasets,
      deleteDataset: mockDeleteDataset,
      generateEmbeddings: mockGenerateEmbeddings,
      clearError: mockClearError,
      // Add other required store properties with default values
      selectedDataset: null,
      datasetMetadata: [],
      setDatasets: jest.fn(),
      addDataset: jest.fn(),
      updateDataset: jest.fn(),
      removeDataset: jest.fn(),
      setSelectedDataset: jest.fn(),
      setDatasetMetadata: jest.fn(),
      setLoading: jest.fn(),
      setError: jest.fn(),
      setEmbeddingGenerationStatus: jest.fn(),
      createDataset: jest.fn(),
      loadDataset: jest.fn(),
      updateDatasetData: jest.fn(),
      addClass: jest.fn(),
      updateClass: jest.fn(),
      removeClass: jest.fn(),
      reset: jest.fn()
    });

    render(
      <DatasetList
        onSelectDataset={mockOnSelectDataset}
        onCreateNew={mockOnCreateNew}
      />
    );

    expect(screen.getByText('Generating...')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
    expect(screen.getByText('Processing example texts...')).toBeInTheDocument();
  });

  test('calls generateEmbeddings when generate button is clicked', async () => {
    render(
      <DatasetList
        onSelectDataset={mockOnSelectDataset}
        onCreateNew={mockOnCreateNew}
      />
    );

    const generateButton = screen.getByText('Generate');
    fireEvent.click(generateButton);

    await waitFor(() => {
      expect(mockGenerateEmbeddings).toHaveBeenCalledWith('test-dataset');
    });
  });

  test('shows embeddings ready status for datasets with embeddings', () => {
    const datasetWithEmbeddings = {
      ...mockDataset,
      embeddingsGenerated: true
    };

    mockUseDatasetStore.mockReturnValue({
      datasets: [datasetWithEmbeddings],
      isLoading: false,
      error: null,
      embeddingGenerationStatus: {},
      loadAllDatasets: mockLoadAllDatasets,
      deleteDataset: mockDeleteDataset,
      generateEmbeddings: mockGenerateEmbeddings,
      clearError: mockClearError,
      // Add other required store properties with default values
      selectedDataset: null,
      datasetMetadata: [],
      setDatasets: jest.fn(),
      addDataset: jest.fn(),
      updateDataset: jest.fn(),
      removeDataset: jest.fn(),
      setSelectedDataset: jest.fn(),
      setDatasetMetadata: jest.fn(),
      setLoading: jest.fn(),
      setError: jest.fn(),
      setEmbeddingGenerationStatus: jest.fn(),
      createDataset: jest.fn(),
      loadDataset: jest.fn(),
      updateDatasetData: jest.fn(),
      addClass: jest.fn(),
      updateClass: jest.fn(),
      removeClass: jest.fn(),
      reset: jest.fn()
    });

    render(
      <DatasetList
        onSelectDataset={mockOnSelectDataset}
        onCreateNew={mockOnCreateNew}
      />
    );

    expect(screen.getByText('Embeddings Ready')).toBeInTheDocument();
    expect(screen.queryByText('Generate')).not.toBeInTheDocument();
  });

  test('loads datasets on mount', () => {
    render(
      <DatasetList
        onSelectDataset={mockOnSelectDataset}
        onCreateNew={mockOnCreateNew}
      />
    );

    expect(mockLoadAllDatasets).toHaveBeenCalled();
  });
});