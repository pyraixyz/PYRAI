import { createBrowserRouter } from 'react-router-dom';
import Layout from '@/components/Layout';
import Dashboard from '@/pages/Dashboard';
import Jobs from '@/pages/Jobs';
import JobDetails from '@/pages/JobDetails';
import SystemMetrics from '@/pages/SystemMetrics';
import Models from '@/pages/Models';
import ModelDetails from '@/pages/ModelDetails';
import ModelTraining from '@/pages/ModelTraining';
import ModelEvaluation from '@/pages/ModelEvaluation';
import Datasets from '@/pages/Datasets';
import DatasetDetails from '@/pages/DatasetDetails';
import DatasetPreprocessing from '@/pages/DatasetPreprocessing';
import Settings from '@/pages/Settings';
import Users from '@/pages/Users';
import Help from '@/pages/Help';

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
        path: '/models/:modelId',
        element: <ModelDetails />,
      },
      {
        path: '/models/:modelId/training',
        element: <ModelTraining />,
      },
      {
        path: '/models/:modelId/evaluation',
        element: <ModelEvaluation />,
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
      {
        path: '/users',
        element: <Users />,
      },
      {
        path: '/help',
        element: <Help />,
      },
    ],
  },
]); 