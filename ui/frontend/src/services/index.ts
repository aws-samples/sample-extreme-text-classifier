/**
 * Service layer exports
 * 
 * This module provides a centralized export point for all API services,
 * making it easy to import and use services throughout the application.
 */

// Core API client
export { ApiClient, apiClient } from './api-client';

// Service implementations
export { ClassificationService, classificationService } from './classification-service';
export { DatasetService, datasetService } from './dataset-service';
export { AttributeService, attributeService } from './attribute-service';
export { FileUploadService, fileUploadService } from './file-upload-service';
export { WizardService, wizardService } from './wizard-service';
export { DatasetFileService } from './dataset-file-service';

// Service types and interfaces
export type {
  FileUploadOptions,
  FileValidationResult,
} from './file-upload-service';

export type {
  WizardGenerationOptions,
  WizardGenerationResult,
} from './wizard-service';

// Re-export API types for convenience
export type {
  ApiResponse,
  ApiError,
  ApiClientConfig,
  RequestOptions,
  UploadProgress,
  DatasetAPI,
  ClassificationAPI,
  AttributeAPI,
  WizardAPI,
} from '../types/api';

/**
 * Service registry for dependency injection or testing
 */
export class ServiceRegistry {
  private static instance: ServiceRegistry;
  private services: Map<string, any> = new Map();

  private constructor() {
    // Register default services
    const { apiClient } = require('./api-client');
    const { classificationService } = require('./classification-service');
    const { datasetService } = require('./dataset-service');
    const { attributeService } = require('./attribute-service');
    const { fileUploadService } = require('./file-upload-service');
    const { wizardService } = require('./wizard-service');
    
    this.register('apiClient', apiClient);
    this.register('classificationService', classificationService);
    this.register('datasetService', datasetService);
    this.register('attributeService', attributeService);
    this.register('fileUploadService', fileUploadService);
    this.register('wizardService', wizardService);
  }

  static getInstance(): ServiceRegistry {
    if (!ServiceRegistry.instance) {
      ServiceRegistry.instance = new ServiceRegistry();
    }
    return ServiceRegistry.instance;
  }

  register<T>(name: string, service: T): void {
    this.services.set(name, service);
  }

  get<T>(name: string): T {
    const service = this.services.get(name);
    if (!service) {
      throw new Error(`Service '${name}' not found in registry`);
    }
    return service as T;
  }

  has(name: string): boolean {
    return this.services.has(name);
  }

  unregister(name: string): boolean {
    return this.services.delete(name);
  }

  clear(): void {
    this.services.clear();
  }

  getAll(): Record<string, any> {
    return Object.fromEntries(this.services.entries());
  }
}

/**
 * Get the service registry instance
 */
export const serviceRegistry = ServiceRegistry.getInstance();

/**
 * Utility function to get a service from the registry
 */
export function getService<T>(name: string): T {
  return serviceRegistry.get<T>(name);
}

/**
 * Utility function to check if all services are available
 */
export function checkServiceHealth(): {
  healthy: boolean;
  services: Record<string, boolean>;
} {
  const services = {
    apiClient: serviceRegistry.has('apiClient'),
    classificationService: serviceRegistry.has('classificationService'),
    datasetService: serviceRegistry.has('datasetService'),
    attributeService: serviceRegistry.has('attributeService'),
    fileUploadService: serviceRegistry.has('fileUploadService'),
    wizardService: serviceRegistry.has('wizardService'),
  };

  const healthy = Object.values(services).every(available => available);

  return { healthy, services };
}

/**
 * Initialize services with custom configuration
 */
export function initializeServices(config: {
  apiBaseUrl?: string;
  timeout?: number;
  retryAttempts?: number;
} = {}): void {
  // Import services dynamically
  const { apiClient } = require('./api-client');
  const { classificationService } = require('./classification-service');
  const { datasetService } = require('./dataset-service');
  const { attributeService } = require('./attribute-service');
  const { fileUploadService } = require('./file-upload-service');
  const { wizardService } = require('./wizard-service');

  // Update API client configuration if provided
  if (config.apiBaseUrl) {
    apiClient.setBaseUrl(config.apiBaseUrl);
  }

  // Re-register services with updated configuration
  serviceRegistry.register('apiClient', apiClient);
  serviceRegistry.register('classificationService', classificationService);
  serviceRegistry.register('datasetService', datasetService);
  serviceRegistry.register('attributeService', attributeService);
  serviceRegistry.register('fileUploadService', fileUploadService);
  serviceRegistry.register('wizardService', wizardService);
}

/**
 * Service configuration for different environments
 */
export const serviceConfigs = {
  development: {
    apiBaseUrl: 'http://localhost:8000',
    timeout: 30000,
    retryAttempts: 3,
  },
  production: {
    apiBaseUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',
    timeout: 60000,
    retryAttempts: 2,
  },
  test: {
    apiBaseUrl: 'http://localhost:8001',
    timeout: 10000,
    retryAttempts: 1,
  },
};

/**
 * Initialize services for the current environment
 */
export function initializeForEnvironment(
  environment: keyof typeof serviceConfigs = 'development'
): void {
  const config = serviceConfigs[environment];
  initializeServices(config);
}

// Auto-initialize for development if in browser environment
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  initializeForEnvironment('development');
}