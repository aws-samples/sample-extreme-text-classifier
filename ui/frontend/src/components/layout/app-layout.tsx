import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from '../navigation/sidebar';

export const AppLayout: React.FC = () => {
  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
};