/**
 * Dataset-related TypeScript interfaces and types
 */

export interface ClassDefinition {
  id: string;
  name: string;
  description: string;
  examples?: string[];
  metadata?: Record<string, any>;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  classes: ClassDefinition[];
  embeddingsGenerated: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface DatasetMetadata {
  id: string;
  name: string;
  description: string;
  classCount: number;
  embeddingsGenerated: boolean;
  createdAt: Date;
  updatedAt: Date;
  fileSize?: number;
}

export interface CreateDatasetRequest {
  name: string;
  description: string;
  classes: Omit<ClassDefinition, 'id'>[];
}

export interface UpdateDatasetRequest {
  id: string;
  name?: string;
  description?: string;
  classes?: ClassDefinition[];
}