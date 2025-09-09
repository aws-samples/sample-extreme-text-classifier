import React, { useState, useEffect } from 'react';
import { Dataset } from '../types/dataset';
import { DatasetList, DatasetEditor } from '../components/dataset';
import { useDatasetStore } from '../stores/dataset-store';

type ViewMode = 'list' | 'create' | 'edit';

export const Datasets: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const { setDatasets } = useDatasetStore();

  useEffect(() => {
    // Load existing datasets from the datasets directory
    loadExistingDatasets();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadExistingDatasets = async () => {
    try {
      // Load datasets from the frontend/datasets directory
      const datasetFiles = ['medical_supplies.json', 'sample_documents.json'];
      const loadedDatasets: Dataset[] = [];

      for (const filename of datasetFiles) {
        try {
          const response = await fetch(`/datasets/${filename}`);
          if (response.ok) {
            const datasetData = await response.json();
            
            // Convert the file format to our Dataset interface
            const dataset: Dataset = {
              id: datasetData.id,
              name: datasetData.name,
              description: datasetData.description,
              classes: datasetData.classes.map((cls: any, index: number) => ({
                id: `${datasetData.id}_class_${index}`,
                name: cls.name,
                description: cls.description,
                examples: cls.examples || [],
                metadata: cls.metadata
              })),
              embeddingsGenerated: datasetData.embeddings_generated || false,
              createdAt: new Date(datasetData.created_at),
              updatedAt: new Date(datasetData.updated_at)
            };
            
            loadedDatasets.push(dataset);
          }
        } catch (error) {
          console.warn(`Failed to load dataset ${filename}:`, error);
        }
      }

      if (loadedDatasets.length > 0) {
        setDatasets(loadedDatasets);
      }
    } catch (error) {
      console.error('Failed to load existing datasets:', error);
    }
  };

  const handleSelectDataset = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setViewMode('edit');
  };

  const handleCreateNew = () => {
    setSelectedDataset(null);
    setViewMode('create');
  };

  const handleSave = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setViewMode('list');
  };

  const handleCancel = () => {
    setSelectedDataset(null);
    setViewMode('list');
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Classification Dataset Management</h1>
        <p className="text-gray-600 mt-1">Load, create, and manage classification datasets</p>
      </div>
      
      {viewMode === 'list' && (
        <DatasetList
          onSelectDataset={handleSelectDataset}
          onCreateNew={handleCreateNew}
          selectedDatasetId={selectedDataset?.id}
        />
      )}

      {(viewMode === 'create' || viewMode === 'edit') && (
        <DatasetEditor
          dataset={selectedDataset || undefined}
          onSave={handleSave}
          onCancel={handleCancel}
          mode={viewMode === 'create' ? 'create' : 'edit'}
        />
      )}
    </div>
  );
};