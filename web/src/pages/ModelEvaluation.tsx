import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Stack,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  List,
  ListItem,
  ListItemText,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Snackbar,
  LinearProgress,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  Compare as CompareIcon,
  Assessment as AssessmentIcon,
  Timeline as TimelineIcon,
  Memory as ResourceIcon,
  BugReport as ErrorIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';
import { modelApi } from '@/api/client';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`eval-tabpanel-${index}`}
      aria-labelledby={`eval-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

function PerformanceMetrics({ evaluation }: { evaluation: any }) {
  return (
    <Stack spacing={3}>
      {/* Overall Metrics */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={3}>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Accuracy
            </Typography>
            <Typography variant="h4" color="primary">
              {(evaluation.metrics.accuracy * 100).toFixed(2)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={evaluation.metrics.accuracy * 100}
              sx={{ width: '100%', mt: 1 }}
            />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              F1 Score
            </Typography>
            <Typography variant="h4" color="primary">
              {evaluation.metrics.f1_score.toFixed(3)}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={evaluation.metrics.f1_score * 100}
              sx={{ width: '100%', mt: 1 }}
            />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Precision
            </Typography>
            <Typography variant="h4" color="primary">
              {evaluation.metrics.precision.toFixed(3)}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={evaluation.metrics.precision * 100}
              sx={{ width: '100%', mt: 1 }}
            />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Recall
            </Typography>
            <Typography variant="h4" color="primary">
              {evaluation.metrics.recall.toFixed(3)}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={evaluation.metrics.recall * 100}
              sx={{ width: '100%', mt: 1 }}
            />
          </Paper>
        </Grid>
      </Grid>

      {/* Confusion Matrix */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Confusion Matrix</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell />
                  {evaluation.confusion_matrix.labels.map((label: string) => (
                    <TableCell key={label} align="center">
                      {label}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {evaluation.confusion_matrix.matrix.map(
                  (row: number[], i: number) => (
                    <TableRow key={i}>
                      <TableCell component="th" scope="row">
                        {evaluation.confusion_matrix.labels[i]}
                      </TableCell>
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
                            color: i === j || cell > 0 ? 'white' : 'inherit',
                          }}
                        >
                          {cell}
                        </TableCell>
                      ))}
                    </TableRow>
                  )
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Stack>
      </Paper>

      {/* ROC Curve */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">ROC Curve</Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={evaluation.roc_curve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="fpr"
                  label={{
                    value: 'False Positive Rate',
                    position: 'bottom',
                  }}
                />
                <YAxis
                  label={{
                    value: 'True Positive Rate',
                    angle: -90,
                    position: 'left',
                  }}
                />
                <ChartTooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="tpr"
                  stroke="#8884d8"
                  name="ROC Curve"
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
          <Typography variant="body2" color="text.secondary" align="center">
            AUC: {evaluation.metrics.auc.toFixed(3)}
          </Typography>
        </Stack>
      </Paper>

      {/* Per-Class Metrics */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Per-Class Performance</Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={evaluation.class_metrics}>
                <PolarGrid />
                <PolarAngleAxis dataKey="class" />
                <PolarRadiusAxis angle={30} domain={[0, 1]} />
                <Radar
                  name="Precision"
                  dataKey="precision"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Radar
                  name="Recall"
                  dataKey="recall"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  fillOpacity={0.6}
                />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </Box>
        </Stack>
      </Paper>
    </Stack>
  );
}

function ResourceUsage({ evaluation }: { evaluation: any }) {
  return (
    <Stack spacing={3}>
      {/* Resource Metrics */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={3}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2" color="text.secondary">
                Inference Time
              </Typography>
              <Typography variant="h4">
                {evaluation.resource_usage.inference_time.toFixed(2)}
                <Typography
                  component="span"
                  variant="body2"
                  color="text.secondary"
                  sx={{ ml: 1 }}
                >
                  ms
                </Typography>
              </Typography>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2" color="text.secondary">
                Memory Usage
              </Typography>
              <Typography variant="h4">
                {(
                  evaluation.resource_usage.memory_usage / (1024 * 1024 * 1024)
                ).toFixed(2)}
                <Typography
                  component="span"
                  variant="body2"
                  color="text.secondary"
                  sx={{ ml: 1 }}
                >
                  GB
                </Typography>
              </Typography>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2" color="text.secondary">
                GPU Memory
              </Typography>
              <Typography variant="h4">
                {(
                  evaluation.resource_usage.gpu_memory / (1024 * 1024 * 1024)
                ).toFixed(2)}
                <Typography
                  component="span"
                  variant="body2"
                  color="text.secondary"
                  sx={{ ml: 1 }}
                >
                  GB
                </Typography>
              </Typography>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2" color="text.secondary">
                CPU Usage
              </Typography>
              <Typography variant="h4">
                {evaluation.resource_usage.cpu_usage.toFixed(1)}
                <Typography
                  component="span"
                  variant="body2"
                  color="text.secondary"
                  sx={{ ml: 1 }}
                >
                  %
                </Typography>
              </Typography>
            </Stack>
          </Paper>
        </Grid>
      </Grid>

      {/* Resource Usage Over Time */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Resource Usage Over Time</Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={evaluation.resource_usage.timeline}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <ChartTooltip />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="memory"
                  stroke="#8884d8"
                  name="Memory (GB)"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="cpu"
                  stroke="#82ca9d"
                  name="CPU (%)"
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </Stack>
      </Paper>

      {/* Batch Processing Analysis */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Batch Processing Analysis</Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={6}>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid />
                    <XAxis
                      type="number"
                      dataKey="batch_size"
                      name="Batch Size"
                      label={{
                        value: 'Batch Size',
                        position: 'bottom',
                      }}
                    />
                    <YAxis
                      type="number"
                      dataKey="time"
                      name="Processing Time"
                      label={{
                        value: 'Processing Time (ms)',
                        angle: -90,
                        position: 'left',
                      }}
                    />
                    <ZAxis
                      type="number"
                      dataKey="memory"
                      range={[50, 400]}
                      name="Memory Usage"
                    />
                    <ChartTooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter
                      name="Batch Analysis"
                      data={evaluation.batch_analysis}
                      fill="#8884d8"
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </Box>
            </Grid>
            <Grid item xs={12} lg={6}>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Batch Size</TableCell>
                      <TableCell align="right">Processing Time (ms)</TableCell>
                      <TableCell align="right">Memory Usage (MB)</TableCell>
                      <TableCell align="right">Throughput (samples/s)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {evaluation.batch_analysis.map((item: any) => (
                      <TableRow key={item.batch_size}>
                        <TableCell>{item.batch_size}</TableCell>
                        <TableCell align="right">
                          {item.time.toFixed(2)}
                        </TableCell>
                        <TableCell align="right">
                          {(item.memory / (1024 * 1024)).toFixed(2)}
                        </TableCell>
                        <TableCell align="right">
                          {((item.batch_size * 1000) / item.time).toFixed(2)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          </Grid>
        </Stack>
      </Paper>
    </Stack>
  );
}

function ErrorAnalysis({ evaluation }: { evaluation: any }) {
  return (
    <Stack spacing={3}>
      {/* Error Distribution */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Error Distribution</Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={evaluation.error_analysis.distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" />
                <YAxis />
                <ChartTooltip />
                <Legend />
                <Bar
                  dataKey="count"
                  fill="#8884d8"
                  name="Number of Errors"
                />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Stack>
      </Paper>

      {/* Common Error Patterns */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Common Error Patterns</Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Pattern</TableCell>
                  <TableCell>Description</TableCell>
                  <TableCell align="right">Frequency</TableCell>
                  <TableCell>Suggested Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {evaluation.error_analysis.patterns.map((pattern: any) => (
                  <TableRow key={pattern.id}>
                    <TableCell>{pattern.name}</TableCell>
                    <TableCell>{pattern.description}</TableCell>
                    <TableCell align="right">
                      <Chip
                        label={`${pattern.frequency}%`}
                        color={
                          pattern.frequency > 10
                            ? 'error'
                            : pattern.frequency > 5
                            ? 'warning'
                            : 'success'
                        }
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{pattern.suggestion}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Stack>
      </Paper>

      {/* Edge Cases */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Edge Cases</Typography>
          <Grid container spacing={2}>
            {evaluation.error_analysis.edge_cases.map((edge: any) => (
              <Grid item xs={12} md={6} key={edge.id}>
                <Paper
                  variant="outlined"
                  sx={{ p: 2, bgcolor: 'background.default' }}
                >
                  <Stack spacing={1}>
                    <Typography variant="subtitle2">{edge.type}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {edge.description}
                    </Typography>
                    <Box>
                      <Chip
                        label={`Impact: ${edge.impact}%`}
                        color={
                          edge.impact > 10
                            ? 'error'
                            : edge.impact > 5
                            ? 'warning'
                            : 'success'
                        }
                        size="small"
                        sx={{ mr: 1 }}
                      />
                      <Chip
                        label={edge.status}
                        variant="outlined"
                        size="small"
                      />
                    </Box>
                  </Stack>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Stack>
      </Paper>
    </Stack>
  );
}

export default function ModelEvaluation() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const queryClient = useQueryClient();

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch evaluation results
  const { data: evaluation, isLoading } = useQuery({
    queryKey: ['model', modelId, 'evaluation'],
    queryFn: () => modelApi.getEvaluation(modelId!),
    enabled: !!modelId,
  });

  // Start evaluation mutation
  const { mutate: startEvaluation } = useMutation({
    mutationFn: () => modelApi.startEvaluation(modelId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'evaluation']);
      setMessage('Evaluation started successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to start evaluation');
      setSeverity('error');
    },
  });

  // Export results mutation
  const { mutate: exportResults } = useMutation({
    mutationFn: () => modelApi.exportEvaluation(modelId!),
    onSuccess: () => {
      setMessage('Results exported successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to export results');
      setSeverity('error');
    },
  });

  if (!model || !evaluation) {
    return null;
  }

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          mb: 3,
        }}
      >
        <Box>
          <Typography variant="h5" gutterBottom>
            Model Evaluation
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <Typography variant="body2" color="text.secondary">
              Last evaluated: {format(new Date(evaluation.timestamp), 'PPp')}
            </Typography>
            <Chip
              label={evaluation.status}
              color={
                evaluation.status === 'completed'
                  ? 'success'
                  : evaluation.status === 'running'
                  ? 'warning'
                  : 'error'
              }
              size="small"
            />
          </Stack>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => startEvaluation()}
          >
            Re-evaluate
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={() => exportResults()}
          >
            Export Results
          </Button>
          <Tooltip title="Share Results">
            <IconButton>
              <ShareIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Compare Models">
            <IconButton>
              <CompareIcon />
            </IconButton>
          </Tooltip>
        </Stack>
      </Box>

      {/* Content */}
      <Card>
        <CardContent>
          <Tabs
            value={tabValue}
            onChange={(_, newValue) => setTabValue(newValue)}
            aria-label="evaluation tabs"
          >
            <Tab
              icon={<AssessmentIcon />}
              label="Performance"
              id="eval-tab-0"
              aria-controls="eval-tabpanel-0"
            />
            <Tab
              icon={<ResourceIcon />}
              label="Resource Usage"
              id="eval-tab-1"
              aria-controls="eval-tabpanel-1"
            />
            <Tab
              icon={<ErrorIcon />}
              label="Error Analysis"
              id="eval-tab-2"
              aria-controls="eval-tabpanel-2"
            />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <PerformanceMetrics evaluation={evaluation} />
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <ResourceUsage evaluation={evaluation} />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <ErrorAnalysis evaluation={evaluation} />
          </TabPanel>
        </CardContent>
      </Card>

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