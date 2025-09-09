import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input, Textarea } from '../ui/input';
import { Dataset, ClassDefinition, CreateDatasetRequest, UpdateDatasetRequest } from '../../types/dataset';
import { useDatasetStore } from '../../stores/dataset-store';
import { ClassCard } from './ClassCard';
import { Plus, Save, X, ArrowLeft } from 'lucide-react';

interface DatasetEditorProps {
  dataset?: Dataset;
  onSave: (dataset: Dataset) => void;
  onCancel: () => void;
  mode: 'create' | 'edit';
}

export const DatasetEditor: React.FC<DatasetEditorProps> = ({
  dataset,
  onSave,
  onCancel,
  mode
}) => {
  const {
    createDataset,
    updateDatasetData,
    generateEmbeddings,
    isLoading,
    error,
    clearError
  } = useDatasetStore();

  const [formData, setFormData] = useState({
    name: dataset?.name || '',
    description: dataset?.description || ''
  });

  const [classes, setClasses] = useState<ClassDefinition[]>(dataset?.classes || []);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isAutoSaving, setIsAutoSaving] = useState(false);

  useEffect(() => {
    if (dataset) {
      setFormData({
        name: dataset.name,
        description: dataset.description
      });
      setClasses(dataset.classes);
    }
  }, [dataset]);

  const validateForm = () => {
    const newErrors: Record<string, string> = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Dataset name is required';
    }

    if (!formData.description.trim()) {
      newErrors.description = 'Dataset description is required';
    }

    if (classes.length === 0) {
      newErrors.classes = 'At least one class is required';
    }

    // Validate class names are unique
    const classNames = classes.map(c => c.name.toLowerCase());
    const duplicateNames = classNames.filter((name, index) => classNames.indexOf(name) !== index);
    if (duplicateNames.length > 0) {
      newErrors.classes = 'Class names must be unique';
    }

    // Validate each class
    classes.forEach((cls, index) => {
      if (!cls.name.trim()) {
        newErrors[`class_${index}_name`] = 'Class name is required';
      }
      if (!cls.description.trim()) {
        newErrors[`class_${index}_description`] = 'Class description is required';
      }
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSave = async () => {
    if (!validateForm()) {
      return;
    }

    clearError();

    try {
      let savedDataset: Dataset;

      if (mode === 'create') {
        const request: CreateDatasetRequest = {
          name: formData.name,
          description: formData.description,
          classes: classes.map(({ id, ...cls }) => cls)
        };
        savedDataset = await createDataset(request);
        
        // Auto-generate embeddings for new datasets
        setTimeout(() => {
          generateEmbeddings(savedDataset.id).catch(error => {
            console.warn('Failed to auto-generate embeddings:', error);
          });
        }, 500);
      } else {
        const request: UpdateDatasetRequest = {
          id: dataset!.id,
          name: formData.name,
          description: formData.description,
          classes: classes
        };
        savedDataset = await updateDatasetData(request);
        
        // Regenerate embeddings if classes were modified
        const classesChanged = JSON.stringify(dataset!.classes) !== JSON.stringify(classes);
        if (classesChanged) {
          setTimeout(() => {
            generateEmbeddings(savedDataset.id).catch(error => {
              console.warn('Failed to regenerate embeddings:', error);
            });
          }, 500);
        }
      }

      onSave(savedDataset);
    } catch (error) {
      console.error('Failed to save dataset:', error);
    }
  };

  const handleAddClass = () => {
    const newClass: ClassDefinition = {
      id: `temp_${Date.now()}`,
      name: '',
      description: '',
      examples: []
    };
    // Add new class at the beginning of the list so it appears right after the "Add Class" button
    setClasses([newClass, ...classes]);
  };

  const handleUpdateClass = (classId: string, updatedClass: Partial<ClassDefinition>) => {
    const updatedClasses = classes.map(cls => 
      cls.id === classId ? { ...cls, ...updatedClass } : cls
    );
    setClasses(updatedClasses);
    
    // Clear related errors
    setErrors(prev => {
      const newErrors = { ...prev };
      const classIndex = classes.findIndex(c => c.id === classId);
      delete newErrors[`class_${classIndex}_name`];
      delete newErrors[`class_${classIndex}_description`];
      return newErrors;
    });

    // Auto-save changes if we're in edit mode and the dataset exists
    if (mode === 'edit' && dataset) {
      // Debounced save to avoid too many API calls
      clearTimeout((window as any).autoSaveTimeout);
      setIsAutoSaving(true);
      
      (window as any).autoSaveTimeout = setTimeout(async () => {
        try {
          await updateDatasetData({
            id: dataset.id,
            name: formData.name,
            description: formData.description,
            classes: updatedClasses
          });
        } catch (error) {
          console.warn('Auto-save failed:', error);
        } finally {
          setIsAutoSaving(false);
        }
      }, 1000);
    }
  };

  const handleRemoveClass = (classId: string) => {
    if (classes.length <= 1) {
      setErrors({ ...errors, classes: 'At least one class is required' });
      return;
    }
    
    const updatedClasses = classes.filter(cls => cls.id !== classId);
    setClasses(updatedClasses);
    setErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors.classes;
      return newErrors;
    });

    // Auto-save changes if we're in edit mode and the dataset exists
    if (mode === 'edit' && dataset) {
      setIsAutoSaving(true);
      
      // Immediate save for class removal
      setTimeout(async () => {
        try {
          await updateDatasetData({
            id: dataset.id,
            name: formData.name,
            description: formData.description,
            classes: updatedClasses
          });
        } catch (error) {
          console.warn('Auto-save failed:', error);
        } finally {
          setIsAutoSaving(false);
        }
      }, 100);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={onCancel}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h2 className="text-xl font-semibold text-gray-900">
              {mode === 'create' ? 'Create New Dataset' : 'Edit Dataset'}
            </h2>
            <div className="flex items-center gap-2 mt-1">
              <p className="text-sm text-gray-600">
                {mode === 'create' 
                  ? 'Define classes and examples for your classification dataset'
                  : 'Modify dataset information and classes'
                }
              </p>
              {isAutoSaving && (
                <div className="flex items-center gap-1 text-xs text-blue-600">
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
                  <span>Saving...</span>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={onCancel}>
            <X className="h-4 w-4 mr-2" />
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={isLoading}>
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {mode === 'create' ? 'Create Dataset' : 'Save Changes'}
          </Button>
        </div>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4">
            <div className="text-red-600">
              <h4 className="font-semibold">Error saving dataset</h4>
              <p className="text-sm mt-1">{error.message}</p>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Dataset Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dataset Name *
            </label>
            <Input
              value={formData.name}
              onChange={(e) => {
                setFormData({ ...formData, name: e.target.value });
                if (errors.name) {
                  setErrors({ ...errors, name: '' });
                }
              }}
              placeholder="Enter dataset name"
              className={errors.name ? 'border-red-300' : ''}
            />
            {errors.name && (
              <p className="text-red-600 text-sm mt-1">{errors.name}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description *
            </label>
            <Textarea
              value={formData.description}
              onChange={(e) => {
                setFormData({ ...formData, description: e.target.value });
                if (errors.description) {
                  setErrors({ ...errors, description: '' });
                }
              }}
              placeholder="Describe what this dataset is used for"
              rows={3}
              className={errors.description ? 'border-red-300' : ''}
            />
            {errors.description && (
              <p className="text-red-600 text-sm mt-1">{errors.description}</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle>Classification Classes</CardTitle>
            <p className="text-sm text-gray-600 mt-1">
              Define the classes that documents can be classified into
            </p>
          </div>
        </CardHeader>
        <CardContent>
          {errors.classes && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-red-600 text-sm">{errors.classes}</p>
            </div>
          )}
          
          {/* Add Class Button - positioned at the top of the classes list */}
          <div className="mb-4">
            <Button onClick={handleAddClass} size="sm" variant="outline" className="w-full border-dashed">
              <Plus className="h-4 w-4 mr-2" />
              Add New Class
            </Button>
          </div>
          
          {classes.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No classes defined yet. Click "Add New Class" above to get started.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {classes.map((cls, index) => (
                <ClassCard
                  key={cls.id}
                  classDefinition={cls}
                  onUpdate={(updatedClass) => handleUpdateClass(cls.id, updatedClass)}
                  onRemove={() => handleRemoveClass(cls.id)}
                  canRemove={classes.length > 1}
                  errors={{
                    name: errors[`class_${index}_name`],
                    description: errors[`class_${index}_description`]
                  }}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};