/**
 * File upload utilities for PDF processing and other file operations
 */

import { apiClient } from './api-client';
import { ApiResponse, UploadProgress } from '../types/api';
import { PDFExtractionResult } from '../types/classification';
import { ErrorHandler } from '../utils/error-handler';

export interface FileUploadOptions {
  onProgress?: (progress: UploadProgress) => void;
  maxSize?: number;
  allowedTypes?: string[];
  timeout?: number;
}

export interface FileValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

export class FileUploadService {
  private readonly defaultMaxSize = 50 * 1024 * 1024; // 50MB
  private readonly defaultAllowedTypes = ['.pdf'];
  private readonly defaultTimeout = 60000; // 60 seconds

  /**
   * Upload and extract content from PDF file
   */
  async uploadAndExtractPDF(
    file: File,
    options: FileUploadOptions = {}
  ): Promise<ApiResponse<PDFExtractionResult>> {
    try {
      // Validate file
      const validation = this.validateFile(file, {
        maxSize: options.maxSize || this.defaultMaxSize,
        allowedTypes: options.allowedTypes || ['.pdf'],
      });

      if (!validation.isValid) {
        return {
          success: false,
          error: {
            type: 'validation',
            message: 'File validation failed',
            details: validation.errors.join(', '),
            recoverable: true,
            suggestedAction: 'Please select a valid PDF file and try again.',
          },
          timestamp: new Date().toISOString(),
        };
      }

      // Upload and extract
      const response = await apiClient.uploadFile<PDFExtractionResult>(
        '/extract/pdf',
        file,
        {},
        options.onProgress
      );

      if (!response.success) {
        ErrorHandler.logError(response.error!);
      }

      return response;
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'processing',
          message: 'PDF extraction failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try with a different PDF file.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Upload file for classification
   */
  async uploadForClassification(
    file: File,
    datasetId: string,
    config: any,
    options: FileUploadOptions = {}
  ): Promise<ApiResponse<any>> {
    try {
      // Validate file
      const validation = this.validateFile(file, {
        maxSize: options.maxSize || this.defaultMaxSize,
        allowedTypes: options.allowedTypes || ['.pdf'],
      });

      if (!validation.isValid) {
        return {
          success: false,
          error: {
            type: 'validation',
            message: 'File validation failed',
            details: validation.errors.join(', '),
            recoverable: true,
            suggestedAction: 'Please select a valid file and try again.',
          },
          timestamp: new Date().toISOString(),
        };
      }

      // Upload for classification
      const response = await apiClient.uploadFile(
        '/classify/pdf',
        file,
        {
          dataset_id: datasetId,
          config: JSON.stringify(config),
        },
        options.onProgress
      );

      if (!response.success) {
        ErrorHandler.logError(response.error!);
      }

      return response;
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      ErrorHandler.logError(appError);
      
      return {
        success: false,
        error: {
          type: 'processing',
          message: 'File classification failed',
          details: appError.message,
          recoverable: true,
          suggestedAction: 'Please try with a different file.',
        },
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Validate file before upload
   */
  validateFile(
    file: File,
    options: {
      maxSize?: number;
      allowedTypes?: string[];
      minSize?: number;
    } = {}
  ): FileValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    const maxSize = options.maxSize || this.defaultMaxSize;
    const allowedTypes = options.allowedTypes || this.defaultAllowedTypes;
    const minSize = options.minSize || 1;

    // Check if file exists
    if (!file) {
      errors.push('No file selected');
      return { isValid: false, errors, warnings };
    }

    // Check file size
    if (file.size === 0) {
      errors.push('File is empty');
    } else if (file.size < minSize) {
      errors.push(`File is too small (minimum ${this.formatFileSize(minSize)})`);
    } else if (file.size > maxSize) {
      errors.push(`File is too large (maximum ${this.formatFileSize(maxSize)})`);
    }

    // Check file type
    const fileExtension = this.getFileExtension(file.name);
    if (!allowedTypes.some(type => fileExtension.toLowerCase() === type.toLowerCase())) {
      errors.push(`File type not supported. Allowed types: ${allowedTypes.join(', ')}`);
    }

    // Check file name
    if (!file.name || file.name.trim().length === 0) {
      errors.push('File must have a name');
    } else if (file.name.length > 255) {
      errors.push('File name is too long (maximum 255 characters)');
    }

    // Warnings for large files
    if (file.size > 10 * 1024 * 1024) { // 10MB
      warnings.push('Large file may take longer to process');
    }

    // Warning for special characters in filename
    if (/[<>:"/\\|?*]/.test(file.name)) {
      warnings.push('File name contains special characters that may cause issues');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Get file extension from filename
   */
  getFileExtension(filename: string): string {
    const lastDotIndex = filename.lastIndexOf('.');
    return lastDotIndex !== -1 ? filename.substring(lastDotIndex) : '';
  }

  /**
   * Format file size for display
   */
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Get file type description
   */
  getFileTypeDescription(filename: string): string {
    const extension = this.getFileExtension(filename).toLowerCase();
    
    const typeMap: Record<string, string> = {
      '.pdf': 'PDF Document',
      '.doc': 'Word Document',
      '.docx': 'Word Document',
      '.txt': 'Text File',
      '.rtf': 'Rich Text Format',
      '.odt': 'OpenDocument Text',
    };

    return typeMap[extension] || 'Unknown File Type';
  }

  /**
   * Check if file type is supported for classification
   */
  isFileTypeSupported(filename: string): boolean {
    const extension = this.getFileExtension(filename).toLowerCase();
    return this.defaultAllowedTypes.includes(extension);
  }

  /**
   * Create a file preview object
   */
  createFilePreview(file: File): {
    name: string;
    size: string;
    type: string;
    lastModified: Date;
    isSupported: boolean;
  } {
    return {
      name: file.name,
      size: this.formatFileSize(file.size),
      type: this.getFileTypeDescription(file.name),
      lastModified: new Date(file.lastModified),
      isSupported: this.isFileTypeSupported(file.name),
    };
  }

  /**
   * Read file as text (for small text files)
   */
  async readFileAsText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        resolve(event.target?.result as string);
      };
      
      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };
      
      reader.readAsText(file);
    });
  }

  /**
   * Read file as data URL (for previews)
   */
  async readFileAsDataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        resolve(event.target?.result as string);
      };
      
      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };
      
      reader.readAsDataURL(file);
    });
  }

  /**
   * Create a drag and drop handler
   */
  createDragDropHandler(
    onFilesSelected: (files: File[]) => void,
    options: FileUploadOptions = {}
  ) {
    const allowedTypes = options.allowedTypes || this.defaultAllowedTypes;

    return {
      onDragOver: (event: DragEvent) => {
        event.preventDefault();
        event.stopPropagation();
      },
      
      onDragEnter: (event: DragEvent) => {
        event.preventDefault();
        event.stopPropagation();
      },
      
      onDragLeave: (event: DragEvent) => {
        event.preventDefault();
        event.stopPropagation();
      },
      
      onDrop: (event: DragEvent) => {
        event.preventDefault();
        event.stopPropagation();
        
        const files = Array.from(event.dataTransfer?.files || []);
        const validFiles = files.filter(file => {
          const extension = this.getFileExtension(file.name);
          return allowedTypes.some(type => extension.toLowerCase() === type.toLowerCase());
        });
        
        if (validFiles.length > 0) {
          onFilesSelected(validFiles);
        }
      },
    };
  }

  /**
   * Create file input change handler
   */
  createFileInputHandler(
    onFilesSelected: (files: File[]) => void,
    options: FileUploadOptions = {}
  ) {
    return (event: Event) => {
      const input = event.target as HTMLInputElement;
      const files = Array.from(input.files || []);
      
      if (files.length > 0) {
        onFilesSelected(files);
      }
      
      // Reset input to allow selecting the same file again
      input.value = '';
    };
  }
}

// Default file upload service instance
export const fileUploadService = new FileUploadService();