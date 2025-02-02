import { createBrowserRouter } from 'react-router-dom';
import Layout from '@/components/Layout';
import Dashboard from '@/pages/Dashboard';
import Jobs from '@/pages/Jobs';
import JobDetails from '@/pages/JobDetails';
import SystemMetrics from '@/pages/SystemMetrics';
import Models from '@/pages/Models';
import Datasets from '@/pages/Datasets';
import DatasetDetails from '@/pages/DatasetDetails';
import DatasetPreprocessing from '@/pages/DatasetPreprocessing';
import Settings from '@/pages/Settings';

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
      {
        path: '/datasets',
        element: <Datasets />,
      },
      {
        path: '/datasets/:datasetId',
        element: <DatasetDetails />,
      },
      {
        path: '/datasets/:datasetId/preprocessing',
        element: <DatasetPreprocessing />,
      },
      {
        path: '/settings',
        element: <Settings />,
      },
    ],
  },
]); 