/**
 * ClassificationOptions Component Tests
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ClassificationOptions } from '../ClassificationOptions';
import { ClassificationConfig } from '../../../types/classification';
import { useDatasetStore } from '../../../stores/dataset-store';
import { attributeService } from '../../../services/attribute-service';

// Mock the stores and services
jest.mock('../../../stores/dataset-store');
jest.mock('../../../services/attribute-service');

const mockUseDatasetStore = useDatasetStore as jest.MockedFunction<typeof useDatasetStore>;
const mockAttributeService = attributeService as jest.Mocked<typeof attributeService>;

describe('ClassificationOptions', () => {
  const defaultConfig: ClassificationConfig = {
    useReranking: false,
    rerankingModel: 'us.amazon.nova-lite-v1:0',
    useAttributeValidation: false,
    topKCandidates: 5
  };

  const mockOnConfigChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseDatasetStore.mockReturnValue({
      selectedDataset: {
        id: 'test-dataset',
        name: 'Test Dataset',
        description: 'Test dataset description',
        classes: [],
        createdAt: new Date(),
        updatedAt: new Date(),
        embeddingsGenerated: true
      }
    } as any);
  });

  it('should render classification options', () => {
    mockAttributeService.attributesExist.mockResolvedValue(true);

    render(
      <ClassificationOptions
        config={defaultConfig}
        onConfigChange={mockOnConfigChange}
      />
    );

    expect(screen.getByText('Classification Options')).toBeInTheDocument();
    expect(screen.getByText('LLM Reranking')).toBeInTheDocument();
    expect(screen.getByText('Attribute Validation')).toBeInTheDocument();
  });

  it('should show error message when attributes do not exist and validation is disabled', async () => {
    mockAttributeService.attributesExist.mockResolvedValue(false);

    render(
      <ClassificationOptions
        config={defaultConfig}
        onConfigChange={mockOnConfigChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('No attributes available. Please generate attributes first to enable attribute validation.')).toBeInTheDocument();
    });
  });

  it('should disable attribute validation button when attributes do not exist', async () => {
    mockAttributeService.attributesExist.mockResolvedValue(false);

    render(
      <ClassificationOptions
        config={defaultConfig}
        onConfigChange={mockOnConfigChange}
      />
    );

    await waitFor(() => {
      const attributeButton = screen.getByRole('button', { name: /disabled/i });
      expect(attributeButton).toBeDisabled();
      expect(attributeButton).toHaveAttribute('title', 'No attributes available. Please generate attributes first.');
    });
  });

  it('should enable attribute validation button when attributes exist', async () => {
    mockAttributeService.attributesExist.mockResolvedValue(true);

    render(
      <ClassificationOptions
        config={defaultConfig}
        onConfigChange={mockOnConfigChange}
      />
    );

    await waitFor(() => {
      const attributeButton = screen.getByRole('button', { name: /disabled/i });
      expect(attributeButton).not.toBeDisabled();
    });
  });

  it('should show error message when attribute validation is enabled but attributes do not exist', async () => {
    mockAttributeService.attributesExist.mockResolvedValue(false);

    const configWithValidation = {
      ...defaultConfig,
      useAttributeValidation: true
    };

    render(
      <ClassificationOptions
        config={configWithValidation}
        onConfigChange={mockOnConfigChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('No attributes available.')).toBeInTheDocument();
      expect(screen.getByText('Please generate attributes first before enabling attribute validation.')).toBeInTheDocument();
    });
  });

  it('should automatically disable attribute validation when no dataset is selected', async () => {
    mockUseDatasetStore.mockReturnValue({
      selectedDataset: null
    } as any);

    const configWithValidation = {
      ...defaultConfig,
      useAttributeValidation: true
    };

    render(
      <ClassificationOptions
        config={configWithValidation}
        onConfigChange={mockOnConfigChange}
      />
    );

    await waitFor(() => {
      expect(mockOnConfigChange).toHaveBeenCalledWith({
        ...configWithValidation,
        useAttributeValidation: false
      });
    });
  });

  it('should automatically disable attribute validation when attributes check fails', async () => {
    mockAttributeService.attributesExist.mockRejectedValue(new Error('API Error'));

    const configWithValidation = {
      ...defaultConfig,
      useAttributeValidation: true
    };

    render(
      <ClassificationOptions
        config={configWithValidation}
        onConfigChange={mockOnConfigChange}
      />
    );

    await waitFor(() => {
      expect(mockOnConfigChange).toHaveBeenCalledWith({
        ...configWithValidation,
        useAttributeValidation: false
      });
    });
  });

  it('should not allow enabling attribute validation when attributes do not exist', async () => {
    mockAttributeService.attributesExist.mockResolvedValue(false);

    render(
      <ClassificationOptions
        config={defaultConfig}
        onConfigChange={mockOnConfigChange}
      />
    );

    await waitFor(() => {
      const attributeButton = screen.getByRole('button', { name: /disabled/i });
      fireEvent.click(attributeButton);
      
      // Should not call onConfigChange because attributes don't exist
      expect(mockOnConfigChange).not.toHaveBeenCalled();
    });
  });

  it('should allow enabling attribute validation when attributes exist', async () => {
    mockAttributeService.attributesExist.mockResolvedValue(true);

    render(
      <ClassificationOptions
        config={defaultConfig}
        onConfigChange={mockOnConfigChange}
      />
    );

    await waitFor(() => {
      const attributeButton = screen.getByRole('button', { name: /disabled/i });
      fireEvent.click(attributeButton);
      
      expect(mockOnConfigChange).toHaveBeenCalledWith({
        ...defaultConfig,
        useAttributeValidation: true
      });
    });
  });
});