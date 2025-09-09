import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Dataset } from '../../types/dataset';
import { useDatasetStore } from '../../stores/dataset-store';
import { Plus, Database, Calendar, FileText, Zap, Loader2 } from 'lucide-react';

interface DatasetListProps {
  onSelectDataset: (dataset: Dataset) => void;
  onCreateNew: () => void;
  selectedDatasetId?: string;
}

export const DatasetList: React.FC<DatasetListProps> = ({
  onSelectDataset,
  onCreateNew,
  selectedDatasetId
}) => {
  const {
    datasets,
    isLoading,
    error,
    embeddingGenerationStatus,
    loadAllDatasets,
    deleteDataset,
    generateEmbeddings,
    clearError
  } = useDatasetStore();

  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    loadAllDatasets();
  }, [loadAllDatasets]);

  const handleDelete = async (datasetId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (window.confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
      setDeletingId(datasetId);
      try {
        await deleteDataset(datasetId);
      } catch (error) {
        console.error('Failed to delete dataset:', error);
      } finally {
        setDeletingId(null);
      }
    }
  };

  const handleGenerateEmbeddings = async (datasetId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    try {
      await generateEmbeddings(datasetId);
    } catch (error) {
      console.error('Failed to generate embeddings:', error);
    }
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  if (error) {
    return (
      <Card className="border-red-200">
        <CardContent className="p-6">
          <div className="text-red-600 mb-4">
            <h3 className="font-semibold">Error loading datasets</h3>
            <p className="text-sm mt-1">{error.message}</p>
          </div>
          <Button onClick={clearError} variant="outline" size="sm">
            Try Again
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Available Datasets</h2>
          <p className="text-sm text-gray-600 mt-1">
            {datasets.length} dataset{datasets.length !== 1 ? 's' : ''} available
          </p>
        </div>
        <Button onClick={onCreateNew} className="flex items-center gap-2">
          <Plus className="h-4 w-4" />
          Create New Dataset
        </Button>
      </div>

      {isLoading ? (
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-3 text-gray-600">Loading datasets...</span>
            </div>
          </CardContent>
        </Card>
      ) : datasets.length === 0 ? (
        <Card>
          <CardContent className="p-6 text-center">
            <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No datasets found</h3>
            <p className="text-gray-600 mb-4">
              Get started by creating your first classification dataset
            </p>
            <Button onClick={onCreateNew}>Create Your First Dataset</Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {datasets.map((dataset) => {
            const embeddingStatus = embeddingGenerationStatus[dataset.id];
            const isGeneratingEmbeddings = embeddingStatus?.isGenerating || false;
            
            return (
              <Card
                key={dataset.id}
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedDatasetId === dataset.id
                    ? 'ring-2 ring-blue-500 border-blue-200'
                    : 'hover:border-gray-300'
                } ${isGeneratingEmbeddings ? 'opacity-90' : ''}`}
                onClick={() => !isGeneratingEmbeddings && onSelectDataset(dataset)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle className="text-lg">{dataset.name}</CardTitle>
                      <p className="text-sm text-gray-600 mt-1">{dataset.description}</p>
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      {isGeneratingEmbeddings ? (
                        <Badge variant="secondary" className="flex items-center gap-1">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Generating...
                        </Badge>
                      ) : (
                        <Badge variant={dataset.embeddingsGenerated ? 'default' : 'secondary'}>
                          {dataset.embeddingsGenerated ? 'Embeddings Ready' : 'Embeddings Pending'}
                        </Badge>
                      )}
                      
                      {!dataset.embeddingsGenerated && !isGeneratingEmbeddings && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={(e) => handleGenerateEmbeddings(dataset.id, e)}
                          className="flex items-center gap-1 text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                        >
                          <Zap className="h-3 w-3" />
                          Generate
                        </Button>
                      )}
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => handleDelete(dataset.id, e)}
                        disabled={deletingId === dataset.id || isGeneratingEmbeddings}
                        className="text-red-600 hover:text-red-700 hover:bg-red-50"
                      >
                        {deletingId === dataset.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          'Delete'
                        )}
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  {/* Embedding generation progress */}
                  {isGeneratingEmbeddings && embeddingStatus && (
                    <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-blue-900">
                          Generating Embeddings
                        </span>
                        <span className="text-sm text-blue-700">
                          {embeddingStatus.progress}%
                        </span>
                      </div>
                      <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${embeddingStatus.progress}%` }}
                        />
                      </div>
                      <p className="text-xs text-blue-700">{embeddingStatus.message}</p>
                    </div>
                  )}
                  
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-1">
                        <FileText className="h-4 w-4" />
                        <span>{dataset.classes.length} classes</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="h-4 w-4" />
                        <span>Updated {formatDate(dataset.updatedAt)}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};