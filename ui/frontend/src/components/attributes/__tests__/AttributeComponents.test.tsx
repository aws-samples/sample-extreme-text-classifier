import React from 'react';
import { render, screen } from '@testing-library/react';
import { AttributeList } from '../AttributeList';
import { ClassAttribute } from '../../../types/attributes';

// Mock the stores
jest.mock('../../../stores/dataset-store', () => ({
  useDatasetStore: () => ({
    datasets: [],
    selectedDataset: null,
  }),
}));

describe('AttributeList Component', () => {
  const mockAttributes: ClassAttribute[] = [
    {
      classId: 'test-class-1',
      className: 'Test Class 1',
      requiredAttributes: {
        operator: 'AND',
        conditions: [
          {
            id: 'condition-1',
            description: 'Test condition 1',
            type: 'text_match',
            parameters: {}
          }
        ]
      },
      generated: true,
      lastUpdated: new Date('2024-01-01')
    }
  ];

  const mockProps = {
    attributes: mockAttributes,
    isLoading: false,
    onEdit: jest.fn(),
    onDelete: jest.fn(),
    onGenerate: jest.fn(),
    canGenerate: true
  };

  it('renders attribute list with attributes', () => {
    render(<AttributeList {...mockProps} />);
    
    expect(screen.getByText('Class Attributes (1)')).toBeInTheDocument();
    expect(screen.getByText('Test Class 1')).toBeInTheDocument();
    expect(screen.getByText('Generated')).toBeInTheDocument();
  });

  it('renders empty state when no attributes', () => {
    render(<AttributeList {...mockProps} attributes={[]} />);
    
    expect(screen.getByText('No Attributes Found')).toBeInTheDocument();
    expect(screen.getByText('Generate Attributes')).toBeInTheDocument();
  });

  it('renders loading state', () => {
    render(<AttributeList {...mockProps} isLoading={true} />);
    
    // Should show loading skeleton
    expect(document.querySelector('.animate-pulse')).toBeInTheDocument();
  });
});