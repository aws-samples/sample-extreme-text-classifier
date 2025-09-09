import React, { useState } from 'react';
import { Card, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Textarea } from '../ui/input';

import { ClassDefinition } from '../../types/dataset';
import { Trash2, Plus, X, Edit3, Check } from 'lucide-react';

interface ClassCardProps {
  classDefinition: ClassDefinition;
  onUpdate: (updatedClass: Partial<ClassDefinition>) => void;
  onRemove: () => void;
  canRemove: boolean;
  errors?: {
    name?: string;
    description?: string;
  };
}

export const ClassCard: React.FC<ClassCardProps> = ({
  classDefinition,
  onUpdate,
  onRemove,
  canRemove,
  errors = {}
}) => {
  const [isEditing, setIsEditing] = useState(!classDefinition.name?.trim() || !classDefinition.description?.trim());
  const [editData, setEditData] = useState({
    name: classDefinition.name,
    description: classDefinition.description
  });
  const [newExample, setNewExample] = useState('');
  const [isAddingExample, setIsAddingExample] = useState(false);
  const [editingExampleIndex, setEditingExampleIndex] = useState<number | null>(null);
  const [editingExampleText, setEditingExampleText] = useState('');

  const handleSaveEdit = () => {
    if (!editData.name.trim() || !editData.description.trim()) {
      return;
    }

    onUpdate({
      name: editData.name,
      description: editData.description
    });
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    // If this is a new class (empty name and description), remove it entirely
    if (!classDefinition.name?.trim() && !classDefinition.description?.trim()) {
      onRemove();
      return;
    }

    // Otherwise, just reset the form data and exit editing mode
    setEditData({
      name: classDefinition.name,
      description: classDefinition.description
    });
    setIsEditing(false);
  };

  const handleAddExample = () => {
    if (!newExample.trim()) return;

    const currentExamples = classDefinition.examples || [];
    const trimmedExample = newExample.trim();

    // Check for duplicate examples
    if (currentExamples.includes(trimmedExample)) {
      alert('This example already exists for this class.');
      return;
    }

    onUpdate({
      examples: [...currentExamples, trimmedExample]
    });
    setNewExample('');
    setIsAddingExample(false);
  };

  const handleEditExample = (index: number) => {
    const currentExamples = classDefinition.examples || [];
    setEditingExampleIndex(index);
    setEditingExampleText(currentExamples[index]);
  };

  const handleSaveExampleEdit = () => {
    if (!editingExampleText.trim() || editingExampleIndex === null) return;

    const currentExamples = classDefinition.examples || [];
    const trimmedText = editingExampleText.trim();

    // Check for duplicate examples (excluding the current one being edited)
    const otherExamples = currentExamples.filter((_, i) => i !== editingExampleIndex);
    if (otherExamples.includes(trimmedText)) {
      alert('This example already exists for this class.');
      return;
    }

    const updatedExamples = [...currentExamples];
    updatedExamples[editingExampleIndex] = trimmedText;

    onUpdate({
      examples: updatedExamples
    });

    setEditingExampleIndex(null);
    setEditingExampleText('');
  };

  const handleCancelExampleEdit = () => {
    setEditingExampleIndex(null);
    setEditingExampleText('');
  };

  const handleRemoveExample = (index: number) => {
    const currentExamples = classDefinition.examples || [];
    const exampleToRemove = currentExamples[index];

    // Show confirmation for longer examples
    if (exampleToRemove.length > 50) {
      const confirmed = window.confirm(
        `Are you sure you want to remove this example?\n\n"${exampleToRemove.substring(0, 100)}${exampleToRemove.length > 100 ? '...' : ''}"`
      );
      if (!confirmed) return;
    }

    onUpdate({
      examples: currentExamples.filter((_, i) => i !== index)
    });
  };

  return (
    <Card className="border-gray-200" data-testid={`class-card-${classDefinition.id}`}>
      <CardContent className="p-4">
        <div className="space-y-4">
          {/* Class Name and Description */}
          <div className="flex items-start justify-between">
            <div className="flex-1 space-y-3">
              {isEditing ? (
                <>
                  <div>
                    <Input
                      value={editData.name}
                      onChange={(e) => setEditData({ ...editData, name: e.target.value })}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          handleSaveEdit();
                        } else if (e.key === 'Escape') {
                          handleCancelEdit();
                        }
                      }}
                      placeholder="Class name"
                      className={errors.name ? 'border-red-300' : ''}
                      autoFocus={!classDefinition.name}
                    />
                    {errors.name && (
                      <p className="text-red-600 text-sm mt-1">{errors.name}</p>
                    )}
                  </div>
                  <div>
                    <Textarea
                      value={editData.description}
                      onChange={(e) => setEditData({ ...editData, description: e.target.value })}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                          e.preventDefault();
                          handleSaveEdit();
                        } else if (e.key === 'Escape') {
                          handleCancelEdit();
                        }
                      }}
                      placeholder="Class description (Ctrl+Enter to save, Escape to cancel)"
                      rows={2}
                      className={errors.description ? 'border-red-300' : ''}
                    />
                    {errors.description && (
                      <p className="text-red-600 text-sm mt-1">{errors.description}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button size="sm" onClick={handleSaveEdit}>
                      <Check className="h-4 w-4 mr-2" />
                      Save
                    </Button>
                    <Button size="sm" variant="outline" onClick={handleCancelEdit}>
                      <X className="h-4 w-4 mr-2" />
                      Cancel
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-3">
                    <h4 className="font-semibold text-gray-900 text-base">
                      {classDefinition.name || 'Unnamed Class'}
                    </h4>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setIsEditing(true)}
                      className="h-10 w-10 p-0 text-gray-500 hover:text-blue-600 hover:bg-blue-50 hover:border-blue-300 border-gray-200"
                      title="Edit class"
                    >
                      <Edit3 className="h-6 w-6" />
                    </Button>
                  </div>
                  <p className="text-gray-600 text-sm">
                    {classDefinition.description || 'No description provided'}
                  </p>
                </>
              )}
            </div>

            {!isEditing && (
              <Button
                size="sm"
                variant="outline"
                onClick={onRemove}
                disabled={!canRemove}
                className="h-10 w-10 p-0 text-gray-500 hover:text-red-600 hover:bg-red-50 hover:border-red-300 border-gray-200 ml-4 disabled:opacity-50 disabled:cursor-not-allowed"
                title={!canRemove ? 'Cannot remove the last class' : 'Remove class'}
              >
                <Trash2 className="h-6 w-6" />
              </Button>
            )}
          </div>

          {/* Examples Section */}
          {!isEditing && (
            <div className="border-t pt-4">
              <div className="flex items-center justify-between mb-3">
                <h5 className="text-sm font-medium text-gray-700">
                  Examples ({(classDefinition.examples || []).length})
                </h5>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setIsAddingExample(true)}
                  className="h-9 text-sm"
                >
                  <Plus className="h-5 w-5 mr-2" />
                  Add Example
                </Button>
              </div>

              {/* Add Example Input */}
              {isAddingExample && (
                <div className="mb-3 p-3 bg-gray-50 rounded-md">
                  <Textarea
                    value={newExample}
                    onChange={(e) => setNewExample(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                        e.preventDefault();
                        handleAddExample();
                      } else if (e.key === 'Escape') {
                        setIsAddingExample(false);
                        setNewExample('');
                      }
                    }}
                    placeholder="Enter an example text for this class... (Ctrl+Enter to save, Escape to cancel)"
                    rows={2}
                    className="mb-2"
                    autoFocus
                  />
                  <div className="flex items-center gap-2">
                    <Button size="sm" onClick={handleAddExample} disabled={!newExample.trim()}>
                      Add Example
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        setIsAddingExample(false);
                        setNewExample('');
                      }}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              {/* Examples List */}
              {(classDefinition.examples || []).length === 0 ? (
                <p className="text-gray-500 text-sm italic">
                  No examples added yet. Examples help improve classification accuracy.
                </p>
              ) : (
                <div className="space-y-2">
                  {(classDefinition.examples || []).map((example, index) => (
                    <div
                      key={index}
                      className="group flex items-start gap-2 p-3 bg-gray-50 rounded-md hover:bg-gray-100 transition-colors"
                    >
                      {editingExampleIndex === index ? (
                        // Editing mode
                        <div className="flex-1 space-y-2">
                          <Textarea
                            value={editingExampleText}
                            onChange={(e) => setEditingExampleText(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                                e.preventDefault();
                                handleSaveExampleEdit();
                              } else if (e.key === 'Escape') {
                                handleCancelExampleEdit();
                              }
                            }}
                            placeholder="Edit example text... (Ctrl+Enter to save, Escape to cancel)"
                            rows={2}
                            className="text-sm"
                            autoFocus
                          />
                          <div className="flex items-center gap-2">
                            <Button size="sm" onClick={handleSaveExampleEdit} disabled={!editingExampleText.trim()}>
                              <Check className="h-4 w-4 mr-2" />
                              Save
                            </Button>
                            <Button size="sm" variant="outline" onClick={handleCancelExampleEdit}>
                              <X className="h-4 w-4 mr-2" />
                              Cancel
                            </Button>
                          </div>
                        </div>
                      ) : (
                        // Display mode
                        <>
                          <div className="flex-1">
                            <p className="text-sm text-gray-700">{example}</p>
                          </div>
                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleEditExample(index)}
                              className="h-9 w-9 p-0 text-gray-500 hover:text-blue-600 hover:bg-blue-50"
                              title="Edit example"
                            >
                              <Edit3 className="h-5 w-5" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleRemoveExample(index)}
                              className="h-9 w-9 p-0 text-gray-500 hover:text-red-600 hover:bg-red-50"
                              title="Remove example"
                            >
                              <X className="h-5 w-5" />
                            </Button>
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};