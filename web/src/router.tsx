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
import ModelDeployment from '@/pages/ModelDeployment';
import ModelMonitoring from '@/pages/ModelMonitoring';
import ModelVersions from '@/pages/ModelVersions';
import ModelExperiments from '@/pages/ModelExperiments';
import ModelPrediction from '@/pages/ModelPrediction';
import ModelAPI from '@/pages/ModelAPI';
import ModelSettings from '@/pages/ModelSettings';
import ModelPermissions from '@/pages/ModelPermissions';
import ModelAudit from '@/pages/ModelAudit';
import ModelTags from '@/pages/ModelTags';
import ModelDocs from '@/pages/ModelDocs';
import ModelFeedback from '@/pages/ModelFeedback';
import ModelAnalytics from '@/pages/ModelAnalytics';
import ModelComparison from '@/pages/ModelComparison';
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
        path: '/models/:modelId/deployment',
        element: <ModelDeployment />,
      },
      {
        path: '/models/:modelId/monitoring',
        element: <ModelMonitoring />,
      },
      {
        path: '/models/:modelId/versions',
        element: <ModelVersions />,
      },
      {
        path: '/models/:modelId/experiments',
        element: <ModelExperiments />,
      },
      {
        path: '/models/:modelId/prediction',
        element: <ModelPrediction />,
      },
      {
        path: '/models/:modelId/api',
        element: <ModelAPI />,
      },
      {
        path: '/models/:modelId/settings',
        element: <ModelSettings />,
      },
      {
        path: '/models/:modelId/permissions',
        element: <ModelPermissions />,
      },
      {
        path: '/models/:modelId/audit',
        element: <ModelAudit />,
      },
      {
        path: '/models/:modelId/tags',
        element: <ModelTags />,
      },
      {
        path: '/models/:modelId/docs',
        element: <ModelDocs />,
      },
      {
        path: '/models/:modelId/feedback',
        element: <ModelFeedback />,
      },
      {
        path: '/models/:modelId/analytics',
        element: <ModelAnalytics />,
      },
      {
        path: '/models/:modelId/comparison',
        element: <ModelComparison />,
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