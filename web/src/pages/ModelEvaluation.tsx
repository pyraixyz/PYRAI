import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Stack,
  Button,
  Paper,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Snackbar,
  LinearProgress,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Assessment as AssessmentIcon,
  DataUsage as DatasetIcon,
  Speed as PerformanceIcon,
  BugReport as ErrorIcon,
  Download as DownloadIcon,
  BarChart as MetricsIcon,
  Timeline as TrendIcon,
  Visibility as VisualizeIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { modelApi } from '@/api/client';

interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  confusion_matrix: number[][];
  class_metrics: {
    class_name: string;
    precision: number;
    recall: number;
    f1_score: number;
    support: number;
  }[];
  error_analysis: {
    type: string;
    count: number;
    examples: {
      input: string;
      predicted: string;
      actual: string;
      confidence: number;
    }[];
  }[];
  performance_metrics: {
    latency_p50: number;
    latency_p95: number;
    latency_p99: number;
    throughput: number;
    memory_usage: number;
    cpu_usage: number;
  };
}

interface DatasetMetrics {
  total_samples: number;
  class_distribution: {
    class_name: string;
    count: number;
    percentage: number;
  }[];
  feature_statistics: {
    feature_name: string;
    mean: number;
    std: number;
    min: number;
    max: number;
  }[];
}

function MetricCard({ title, value, unit, icon }: any) {
  return (
    <Card>
      <CardContent>
        <Stack direction="row" spacing={2} alignItems="center">
          <Box sx={{ color: 'primary.main' }}>{icon}</Box>
          <Box>
            <Typography variant="h6" gutterBottom>
              {typeof value === 'number' ? value.toFixed(4) : value}
              {unit && <Typography component="span" variant="caption"> {unit}</Typography>}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {title}
            </Typography>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}

export default function ModelEvaluation() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [selectedDataset, setSelectedDataset] = useState<string>('test');
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch evaluation metrics
  const { data: metrics, refetch: refetchMetrics } = useQuery({
    queryKey: ['model', modelId, 'evaluation', selectedDataset],
    queryFn: () => modelApi.getEvaluationMetrics(modelId!, selectedDataset),
    enabled: !!modelId,
  });

  // Fetch dataset metrics
  const { data: datasetMetrics } = useQuery({
    queryKey: ['model', modelId, 'dataset-metrics', selectedDataset],
    queryFn: () => modelApi.getDatasetMetrics(modelId!, selectedDataset),
    enabled: !!modelId,
  });

  // Run evaluation mutation
  const { mutate: runEvaluation, isLoading: isEvaluating } = useMutation({
    mutationFn: () => modelApi.runEvaluation(modelId!, selectedDataset),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'evaluation']);
      setMessage('Evaluation completed successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to run evaluation');
      setSeverity('error');
    },
  });

  if (!model || !metrics || !datasetMetrics) {
    return null;
  }

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Evaluation
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Evaluate performance and quality metrics for {model.name}
        </Typography>
      </Box>

      {/* Controls */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        justifyContent="space-between"
        sx={{ mb: 3 }}
      >
        <Stack direction="row" spacing={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Dataset</InputLabel>
            <Select
              value={selectedDataset}
              label="Dataset"
              onChange={(e) => setSelectedDataset(e.target.value)}
            >
              <MenuItem value="train">Training Set</MenuItem>
              <MenuItem value="validation">Validation Set</MenuItem>
              <MenuItem value="test">Test Set</MenuItem>
            </Select>
          </FormControl>

          <Button
            variant="contained"
            startIcon={<AssessmentIcon />}
            onClick={() => runEvaluation()}
            disabled={isEvaluating}
          >
            {isEvaluating ? 'Evaluating...' : 'Run Evaluation'}
          </Button>
        </Stack>

        <Stack direction="row" spacing={2}>
          <Tooltip title="Refresh">
            <IconButton onClick={() => refetchMetrics()}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Export Results">
            <IconButton>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
        </Stack>
      </Stack>

      {/* Main Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Accuracy"
            value={metrics.accuracy * 100}
            unit="%"
            icon={<MetricsIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="F1 Score"
            value={metrics.f1_score}
            icon={<AssessmentIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="AUC-ROC"
            value={metrics.auc_roc}
            icon={<TrendIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg. Latency"
            value={metrics.performance_metrics.latency_p50}
            unit="ms"
            icon={<PerformanceIcon />}
          />
        </Grid>
      </Grid>

      {/* Detailed Metrics */}
      <Grid container spacing={3}>
        {/* Confusion Matrix */}
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Confusion Matrix
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell />
                    {metrics.confusion_matrix[0].map((_, index) => (
                      <TableCell key={index} align="center">
                        Predicted {index}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {metrics.confusion_matrix.map((row, i) => (
                    <TableRow key={i}>
                      <TableCell>Actual {i}</TableCell>
                      {row.map((cell, j) => (
                        <TableCell
                          key={j}
                          align="center"
                          sx={{
                            bgcolor:
                              i === j
                                ? 'success.light'
                                : cell > 0
                                ? 'error.light'
                                : 'inherit',
                          }}
                        >
                          {cell}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* Class Distribution */}
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Class Distribution
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={datasetMetrics.class_distribution}
                    dataKey="percentage"
                    nameKey="class_name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                  >
                    {datasetMetrics.class_distribution.map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[index % COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Per-Class Metrics */}
        <Grid item xs={12}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Per-Class Metrics
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Class</TableCell>
                    <TableCell align="right">Precision</TableCell>
                    <TableCell align="right">Recall</TableCell>
                    <TableCell align="right">F1 Score</TableCell>
                    <TableCell align="right">Support</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {metrics.class_metrics.map((metric) => (
                    <TableRow key={metric.class_name}>
                      <TableCell>{metric.class_name}</TableCell>
                      <TableCell align="right">
                        {(metric.precision * 100).toFixed(2)}%
                      </TableCell>
                      <TableCell align="right">
                        {(metric.recall * 100).toFixed(2)}%
                      </TableCell>
                      <TableCell align="right">
                        {(metric.f1_score * 100).toFixed(2)}%
                      </TableCell>
                      <TableCell align="right">{metric.support}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* Error Analysis */}
        <Grid item xs={12}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Error Analysis
            </Typography>
            <Grid container spacing={3}>
              {/* Error Distribution */}
              <Grid item xs={12} md={6}>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={metrics.error_analysis}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="type" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Bar dataKey="count" fill="#8884d8" name="Error Count" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </Grid>

              {/* Error Examples */}
              <Grid item xs={12} md={6}>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Input</TableCell>
                        <TableCell>Predicted</TableCell>
                        <TableCell>Actual</TableCell>
                        <TableCell align="right">Confidence</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {metrics.error_analysis
                        .flatMap((error) => error.examples)
                        .slice(0, 5)
                        .map((example, index) => (
                          <TableRow key={index}>
                            <TableCell>{example.input}</TableCell>
                            <TableCell>{example.predicted}</TableCell>
                            <TableCell>{example.actual}</TableCell>
                            <TableCell align="right">
                              {(example.confidence * 100).toFixed(2)}%
                            </TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Performance Metrics
            </Typography>
            <Grid container spacing={3}>
              {/* Latency Distribution */}
              <Grid item xs={12} md={6}>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        {
                          name: 'P50',
                          value: metrics.performance_metrics.latency_p50,
                        },
                        {
                          name: 'P95',
                          value: metrics.performance_metrics.latency_p95,
                        },
                        {
                          name: 'P99',
                          value: metrics.performance_metrics.latency_p99,
                        },
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Bar
                        dataKey="value"
                        fill="#8884d8"
                        name="Latency (ms)"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </Grid>

              {/* Resource Usage */}
              <Grid item xs={12} md={6}>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        {
                          name: 'CPU Usage',
                          value: metrics.performance_metrics.cpu_usage,
                        },
                        {
                          name: 'Memory Usage',
                          value: metrics.performance_metrics.memory_usage,
                        },
                        {
                          name: 'Throughput',
                          value: metrics.performance_metrics.throughput,
                        },
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Bar
                        dataKey="value"
                        fill="#82ca9d"
                        name="Resource Usage"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>

      {/* Message Snackbar */}
      <Snackbar
        open={!!message}
        autoHideDuration={3000}
        onClose={() => setMessage('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setMessage('')}
          severity={severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {message}
        </Alert>
      </Snackbar>
    </Box>
  );
} 