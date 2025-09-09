/**
 * Common utility types
 */

// Generic ID type
export type ID = string;

// Timestamp type
export type Timestamp = string | Date;

// Status types
export type Status = 'idle' | 'loading' | 'success' | 'error';

// Sort order
export type SortOrder = 'asc' | 'desc';

// Generic pagination
export interface Pagination {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
}

// Generic list response
export interface ListResponse<T> {
  items: T[];
  pagination: Pagination;
}

// Generic search params
export interface SearchParams {
  query?: string;
  filters?: Record<string, any>;
  sort?: {
    field: string;
    order: SortOrder;
  };
  pagination?: {
    page: number;
    pageSize: number;
  };
}

// File metadata
export interface FileMetadata {
  name: string;
  size: number;
  type: string;
  lastModified: Date;
  path?: string;
}

// Progress callback
export type ProgressCallback = (progress: number) => void;

// Generic callback
export type Callback<T = void> = (result: T) => void;

// Async callback
export type AsyncCallback<T = void> = (result: T) => Promise<void>;

// Event handler
export type EventHandler<T = any> = (event: T) => void;

// Utility types for making properties optional/required
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

// Deep partial type
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// Exclude null and undefined
export type NonNullable<T> = T extends null | undefined ? never : T;