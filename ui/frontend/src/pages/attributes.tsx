import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { AttributeList, AttributeEditor } from '../components/attributes';
import { useAttributeStore } from '../stores/attribute-store';
import { useDatasetStore } from '../stores/dataset-store';
import { ClassAttribute } from '../types/attributes';
import { AlertCircle, Database, Wand2, RefreshCw } from 'lucide-react';

export const Attributes: React.FC = () => {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const [selectedAttribute, setSelectedAttribute] = useState<ClassAttribute | null>(null);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');

  // Stores
  const {
    isLoading,
    isGenerating,
    error,
    generationProgress,
    loadAttributes,
    generateAttributes,
    saveAttributes,
    getAttributesForDataset,
    hasAttributes,
    clearError
  } = useAttributeStore();

  const {
    datasets,
    selectedDataset,
    setSelectedDataset,
    loadAllDatasets
  } = useDatasetStore();

  // Load datasets on mount
  useEffect(() => {
    loadAllDatasets();
  }, [loadAllDatasets]);

  // Load attributes when dataset is selected
  useEffect(() => {
    if (selectedDatasetId) {
      loadAttributes(selectedDatasetId).catch(console.error);
    }
  }, [selectedDatasetId, loadAttributes]);

  // Set initial dataset selection based on URL parameter or first available
  useEffect(() => {
    if (datasets.length > 0) {
      if (datasetId) {
        // Try to find the dataset by ID from URL parameter
        const targetDataset = datasets.find(d => d.id === datasetId);
        if (targetDataset && targetDataset.id !== selectedDatasetId) {
          setSelectedDatasetId(targetDataset.id);
          setSelectedDataset(targetDataset);
        }
      } else if (!selectedDatasetId) {
        // No URL parameter, select first dataset if none selected
        const firstDataset = datasets[0];
        setSelectedDatasetId(firstDataset.id);
        setSelectedDataset(firstDataset);
      }
    }
  }, [datasetId, datasets, selectedDatasetId, setSelectedDataset]);

  const handleDatasetChange = (datasetId: string) => {
    const dataset = datasets.find(d => d.id === datasetId);
    if (dataset) {
      setSelectedDatasetId(datasetId);
      setSelectedDataset(dataset);
    }
  };

  const handleEditAttribute = (attribute: ClassAttribute) => {
    setSelectedAttribute(attribute);
    setIsEditorOpen(true);
  };

  const handleDeleteAttribute = async (classId: string) => {
    if (!selectedDatasetId) return;
    
    try {
      // Remove the specific attribute from the dataset
      const currentAttributes = getAttributesForDataset(selectedDatasetId);
      const updatedAttributes = currentAttributes.filter(attr => attr.classId !== classId);
      
      await saveAttributes({
        datasetId: selectedDatasetId,
        attributes: updatedAttributes
      });
    } catch (error) {
      console.error('Failed to delete attribute:', error);
    }
  };

  const handleGenerateAttributes = async () => {
    console.log('handleGenerateAttributes called');
    
    if (!selectedDatasetId || !selectedDataset) {
      console.log('Returning early - missing dataset info');
      return;
    }
    
    try {
      console.log('Starting attribute generation for dataset:', selectedDatasetId);
      await generateAttributes({
        datasetId: selectedDatasetId,
        domain: selectedDataset.description || selectedDataset.name,
        classIds: selectedDataset.classes.map(c => c.name)
      });
      console.log('Attribute generation completed, reloading attributes...');
      // Force a refresh of the attributes to ensure UI is updated
      await loadAttributes(selectedDatasetId);
      console.log('Attributes reloaded successfully');
    } catch (error) {
      console.error('Failed to generate attributes:', error);
    }
  };

  const handleGenerateForClass = async (classId: string, className: string) => {
    if (!selectedDatasetId || !selectedDataset) return;
    
    try {
      await generateAttributes({
        datasetId: selectedDatasetId,
        domain: selectedDataset.description || selectedDataset.name,
        classIds: [classId]
      });
      // Force a refresh of the attributes to ensure UI is updated
      await loadAttributes(selectedDatasetId);
    } catch (error) {
      console.error('Failed to generate attributes for class:', error);
    }
  };

  const handleSaveAttribute = async (attribute: ClassAttribute) => {
    if (!selectedDatasetId) return;
    
    try {
      const currentAttributes = getAttributesForDataset(selectedDatasetId);
      console.log('Current attributes before save:', currentAttributes);
      console.log('Saving attribute:', attribute);
      
      const updatedAttributes = currentAttributes.map(attr => 
        attr.classId === attribute.classId ? attribute : attr
      );
      
      // If it's a new attribute, add it
      if (!currentAttributes.find(attr => attr.classId === attribute.classId)) {
        updatedAttributes.push(attribute);
      }
      
      console.log('Updated attributes to save:', updatedAttributes);
      
      await saveAttributes({
        datasetId: selectedDatasetId,
        attributes: updatedAttributes
      });
      
      console.log('Save completed, checking attributes in store...');
      const attributesAfterSave = getAttributesForDataset(selectedDatasetId);
      console.log('Attributes after save:', attributesAfterSave);
      
      setIsEditorOpen(false);
      setSelectedAttribute(null);
    } catch (error) {
      console.error('Failed to save attribute:', error);
    }
  };

  const handleCancelEdit = () => {
    setIsEditorOpen(false);
    setSelectedAttribute(null);
  };

  const currentAttributes = selectedDatasetId ? getAttributesForDataset(selectedDatasetId) : [];
  const hasCurrentAttributes = selectedDatasetId ? hasAttributes(selectedDatasetId) : false;

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Attribute Management</h1>
        <p className="text-gray-600 mt-1">Generate and manage class attributes for enhanced classification</p>
      </div>

      {/* Error Display */}
      {error && (
        <Card className="mb-6 border-red-200 bg-red-50">
          <CardContent className="pt-4">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-red-600 mr-2 mt-0.5" />
              <div className="flex-1">
                <h4 className="text-sm font-medium text-red-800 mb-1">
                  {error.message}
                </h4>
                {error.details && (
                  <p className="text-sm text-red-700">{error.details}</p>
                )}
                {error.suggestedAction && (
                  <p className="text-sm text-red-600 mt-2 font-medium">
                    {error.suggestedAction}
                  </p>
                )}
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearError}
                className="text-red-600 hover:text-red-700"
              >
                Dismiss
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Generation Progress */}
      {generationProgress && (
        <Card className="mb-6 border-blue-200 bg-blue-50">
          <CardContent className="pt-4">
            <div className="flex items-center">
              <Wand2 className="w-5 h-5 text-blue-600 mr-2 animate-spin" />
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-blue-800">
                    Generating Attributes with AI
                  </span>
                  <span className="text-sm text-blue-600">
                    {generationProgress.progress}%
                  </span>
                </div>
                <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${generationProgress.progress}%` }}
                  />
                </div>
                <p className="text-sm text-blue-700 mb-1">{generationProgress.message}</p>
                <p className="text-xs text-blue-600 italic">
                  ⏱️ This process typically takes 3-5 minutes. Please keep this tab open and wait for completion.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Dataset Selection */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Database className="w-5 h-5 mr-2" />
            Dataset Selection
          </CardTitle>
        </CardHeader>
        <CardContent>
          {datasets.length === 0 ? (
            <div className="text-center py-4">
              <p className="text-gray-500 mb-4">No datasets available. Please create a dataset first.</p>
              <Button variant="outline" onClick={() => loadAllDatasets()}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh Datasets
              </Button>
            </div>
          ) : (
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Dataset
                </label>
                <select
                  value={selectedDatasetId}
                  onChange={(e) => handleDatasetChange(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">Choose a dataset...</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.classes.length} classes)
                    </option>
                  ))}
                </select>
              </div>
              {selectedDataset && (
                <div className="text-sm text-gray-600">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">
                      {selectedDataset.classes.length} classes
                    </Badge>
                    <Badge variant={hasCurrentAttributes ? "default" : "secondary"}>
                      {hasCurrentAttributes ? `${currentAttributes.length} attributes` : 'No attributes'}
                    </Badge>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Attributes List */}
      {selectedDatasetId && (
        <AttributeList
          attributes={currentAttributes}
          isLoading={isLoading}
          onEdit={handleEditAttribute}
          onDelete={handleDeleteAttribute}
          onGenerate={handleGenerateAttributes}
          canGenerate={!isGenerating && selectedDataset !== null}
        />
      )}

      {/* Attribute Editor Modal */}
      <AttributeEditor
        attribute={selectedAttribute}
        isOpen={isEditorOpen}
        onSave={handleSaveAttribute}
        onCancel={handleCancelEdit}
        onGenerate={handleGenerateForClass}
        isGenerating={isGenerating}
      />
    </div>
  );
};