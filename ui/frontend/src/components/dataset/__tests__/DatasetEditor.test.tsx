import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DatasetEditor } from '../DatasetEditor';
import { Dataset, ClassDefinition } from '../../../types/dataset';

// Mock the dataset store
jest.mock('../../../stores/dataset-store', () => ({
  useDatasetStore: () => ({
    updateDatasetData: jest.fn().mockResolvedValue({}),
    createDataset: jest.fn().mockResolvedValue({}),
    generateEmbeddings: jest.fn().mockResolvedValue({}),
    isLoading: false,
    error: null,
    clearError: jest.fn(),
  }),
}));

const mockDataset: Dataset = {
  id: 'test-dataset',
  name: 'Test Dataset',
  description: 'Test Description',
  classes: [
    {
      id: 'class-1',
      name: 'Class 1',
      description: 'First class',
      examples: ['Example 1'],
    },
  ],
  embeddingsGenerated: false,
  createdAt: new Date(),
  updatedAt: new Date(),
};

describe('DatasetEditor', () => {
  const mockOnSave = jest.fn();
  const mockOnCancel = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should add new class at the beginning of the list', async () => {
    render(
      <DatasetEditor
        dataset={mockDataset}
        onSave={mockOnSave}
        onCancel={mockOnCancel}
        mode="edit"
      />
    );

    // Find and click the "Add New Class" button
    const addButton = screen.getByText('Add New Class');
    fireEvent.click(addButton);

    // The new class should appear first (before the existing class)
    const classCards = screen.getAllByTestId(/class-card/);
    expect(classCards).toHaveLength(2);
    
    // The first card should be the new empty class
    const firstCard = classCards[0];
    expect(firstCard).toBeInTheDocument();
  });

  it('should show add class button inline with classes', () => {
    render(
      <DatasetEditor
        dataset={mockDataset}
        onSave={mockOnSave}
        onCancel={mockOnCancel}
        mode="edit"
      />
    );

    // The add button should be present and styled as outlined/dashed
    const addButton = screen.getByText('Add New Class');
    expect(addButton).toBeInTheDocument();
    expect(addButton.closest('button')).toHaveClass('border-dashed');
  });

  it('should show auto-save indicator when editing', async () => {
    render(
      <DatasetEditor
        dataset={mockDataset}
        onSave={mockOnSave}
        onCancel={mockOnCancel}
        mode="edit"
      />
    );

    // Add a new class to trigger auto-save
    const addButton = screen.getByText('Add New Class');
    fireEvent.click(addButton);

    // Should show saving indicator
    await waitFor(() => {
      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });
  });

  it('should remove new class when cancel is clicked on empty class', async () => {
    render(
      <DatasetEditor
        dataset={mockDataset}
        onSave={mockOnSave}
        onCancel={mockOnCancel}
        mode="edit"
      />
    );

    // Add a new class
    const addButton = screen.getByText('Add New Class');
    fireEvent.click(addButton);

    // Should have 2 classes now (original + new)
    let classCards = screen.getAllByTestId(/class-card/);
    expect(classCards).toHaveLength(2);

    // Find and click cancel on the new (empty) class
    const cancelButton = screen.getByText('Cancel');
    fireEvent.click(cancelButton);

    // Should be back to 1 class (new class removed)
    await waitFor(() => {
      classCards = screen.getAllByTestId(/class-card/);
      expect(classCards).toHaveLength(1);
    });
  });
});