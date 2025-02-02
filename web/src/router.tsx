import { createBrowserRouter } from 'react-router-dom';
import Layout from '@/components/Layout';
import Dashboard from '@/pages/Dashboard';
import Jobs from '@/pages/Jobs';
import JobDetails from '@/pages/JobDetails';
import SystemMetrics from '@/pages/SystemMetrics';
import Models from '@/pages/Models';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        path: '/',
        element: <Dashboard />,
      },
      {
        path: '/jobs',
        element: <Jobs />,
      },
      {
        path: '/jobs/:jobId',
        element: <JobDetails />,
      },
      {
        path: '/metrics',
        element: <SystemMetrics />,
      },
      {
        path: '/models',
        element: <Models />,
      },
    ],
  },
]); 