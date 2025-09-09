/**
 * Core API client with proper error handling and request management
 */

import { ApiResponse, ApiError, ApiClientConfig, RequestOptions, UploadProgress } from '../types/api';
import { ErrorHandler } from '../utils/error-handler';

export class ApiClient {
  private baseUrl: string;
  private timeout: number;
  private retryAttempts: number;
  private retryDelay: number;

  constructor(config: Partial<ApiClientConfig> = {}) {
    this.baseUrl = config.baseUrl || 'http://localhost:8000';
    this.timeout = config.timeout || 30000;
    this.retryAttempts = config.retryAttempts || 3;
    this.retryDelay = config.retryDelay || 1000;
  }

  /**
   * Make an HTTP request with error handling and retries
   */
  async request<T = any>(
    endpoint: string,
    options: RequestOptions = { method: 'GET' }
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.retryAttempts; attempt++) {
      try {
        const response = await this.makeRequest<T>(url, options);
        return response;
      } catch (error) {
        lastError = error as Error;
        
        // Don't retry on client errors (4xx) or last attempt
        if (attempt === this.retryAttempts || this.isClientError(error)) {
          break;
        }

        // Wait before retrying
        await this.delay(this.retryDelay * Math.pow(2, attempt));
      }
    }

    // Convert error to ApiResponse format
    const apiError = this.handleError(lastError!);
    return {
      success: false,
      error: apiError,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Make the actual HTTP request
   */
  private async makeRequest<T>(url: string, options: RequestOptions): Promise<ApiResponse<T>> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || this.timeout);

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...options.headers,
      };

      let body: any = options.body;
      
      // Handle FormData (for file uploads)
      if (body instanceof FormData) {
        delete headers['Content-Type']; // Let browser set it with boundary
      } else if (body && typeof body === 'object') {
        body = JSON.stringify(body);
      }

      const response = await fetch(url, {
        method: options.method,
        headers,
        body,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle non-JSON responses
      let data: any;
      const contentType = response.headers.get('content-type');
      
      if (contentType?.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${data.error || data.message || data}`);
      }

      // If the response is already in ApiResponse format, return it
      if (data && typeof data === 'object' && 'success' in data) {
        return data;
      }

      // Wrap raw data in ApiResponse format
      return {
        success: true,
        data,
        timestamp: new Date().toISOString(),
      };

    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  /**
   * Upload file with progress tracking
   */
  async uploadFile<T = any>(
    endpoint: string,
    file: File,
    additionalData: Record<string, string> = {},
    onProgress?: (progress: UploadProgress) => void
  ): Promise<ApiResponse<T>> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      
      formData.append('file', file);
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, value);
      });

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress: UploadProgress = {
            loaded: event.loaded,
            total: event.total,
            percentage: Math.round((event.loaded / event.total) * 100),
          };
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        try {
          if (xhr.status >= 200 && xhr.status < 300) {
            const data = JSON.parse(xhr.responseText);
            resolve({
              success: true,
              data,
              timestamp: new Date().toISOString(),
            });
          } else {
            const errorData = JSON.parse(xhr.responseText);
            resolve({
              success: false,
              error: this.handleError(new Error(errorData.error || `HTTP ${xhr.status}`)),
              timestamp: new Date().toISOString(),
            });
          }
        } catch (error) {
          resolve({
            success: false,
            error: this.handleError(error as Error),
            timestamp: new Date().toISOString(),
          });
        }
      });

      xhr.addEventListener('error', () => {
        resolve({
          success: false,
          error: this.handleError(new Error('Upload failed')),
          timestamp: new Date().toISOString(),
        });
      });

      xhr.addEventListener('timeout', () => {
        resolve({
          success: false,
          error: this.handleError(new Error('Upload timeout')),
          timestamp: new Date().toISOString(),
        });
      });

      xhr.timeout = this.timeout;
      xhr.open('POST', `${this.baseUrl}${endpoint}`);
      xhr.send(formData);
    });
  }

  /**
   * GET request helper
   */
  async get<T = any>(endpoint: string, headers?: Record<string, string>): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'GET', headers });
  }

  /**
   * POST request helper
   */
  async post<T = any>(
    endpoint: string,
    body?: any,
    headers?: Record<string, string>
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'POST', body, headers });
  }

  /**
   * PUT request helper
   */
  async put<T = any>(
    endpoint: string,
    body?: any,
    headers?: Record<string, string>
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'PUT', body, headers });
  }

  /**
   * DELETE request helper
   */
  async delete<T = any>(endpoint: string, headers?: Record<string, string>): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'DELETE', headers });
  }

  /**
   * Check if error is a client error (4xx)
   */
  private isClientError(error: any): boolean {
    if (error?.message?.includes('HTTP 4')) return true;
    return false;
  }

  /**
   * Handle and convert errors to ApiError format
   */
  private handleError(error: Error): ApiError {
    // Network errors
    if (error.name === 'AbortError' || error.message.includes('timeout')) {
      return ErrorHandler.createNetworkError('Request timeout', 408).error!;
    }

    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      return ErrorHandler.createNetworkError('Network connection failed').error!;
    }

    // HTTP errors
    const httpMatch = error.message.match(/HTTP (\d+):/);
    if (httpMatch) {
      const statusCode = parseInt(httpMatch[1]);
      return ErrorHandler.createNetworkError(error.message, statusCode).error!;
    }

    // Generic error
    return {
      type: 'network',
      message: error.message || 'Unknown error occurred',
      recoverable: true,
      suggestedAction: 'Please try again or check your connection.',
    };
  }

  /**
   * Delay utility for retries
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Update base URL
   */
  setBaseUrl(baseUrl: string): void {
    this.baseUrl = baseUrl;
  }

  /**
   * Get current configuration
   */
  getConfig(): ApiClientConfig {
    return {
      baseUrl: this.baseUrl,
      timeout: this.timeout,
      retryAttempts: this.retryAttempts,
      retryDelay: this.retryDelay,
    };
  }
}

// Default API client instance
export const apiClient = new ApiClient();