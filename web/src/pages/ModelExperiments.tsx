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
  TextField,
  Alert,
  Snackbar,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Autocomplete,
} from '@mui/material';
import {
  Add as AddIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Compare as CompareIcon,
  Science as ExperimentIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  Timeline as MetricsIcon,
  Settings as ConfigIcon,
  Refresh as RefreshIcon,
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
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';
import { modelApi } from '@/api/client';

interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'completed' | 'failed' | 'stopped';
  config: {
    learning_rate: number;
    batch_size: number;
    epochs: number;
    optimizer: string;
    architecture: string;
    [key: string]: any;
  };
  metrics: {
    accuracy: number;
    loss: number;
    val_accuracy: number;
    val_loss: number;
    [key: string]: number;
  };
  created_at: string;
  updated_at: string;
  created_by: {
    id: string;
    name: string;
  };
}

interface ExperimentDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: any) => void;
  experiment?: Experiment;
}

function ExperimentDialog({
  open,
  onClose,
  onSave,
  experiment,
}: ExperimentDialogProps) {
  const [formData, setFormData] = useState(
    experiment || {
      name: '',
      description: '',
      config: {
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 10,
        optimizer: 'adam',
        architecture: 'resnet50',
      },
    }
  );

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {experiment ? 'Edit Experiment' : 'Create Experiment'}
      </DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <TextField
            label="Name"
            value={formData.name}
            onChange={(e) =>
              setFormData({ ...formData, name: e.target.value })
            }
            fullWidth
            required
          />

          <TextField
            label="Description"
            value={formData.description}
            onChange={(e) =>
              setFormData({ ...formData, description: e.target.value })
            }
            multiline
            rows={3}
            fullWidth
          />

          <Typography variant="subtitle1" gutterBottom>
            Configuration
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Learning Rate"
                type="number"
                value={formData.config.learning_rate}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    config: {
                      ...formData.config,
                      learning_rate: parseFloat(e.target.value),
                    },
                  })
                }
                fullWidth
                inputProps={{ step: 0.0001 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Batch Size"
                type="number"
                value={formData.config.batch_size}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    config: {
                      ...formData.config,
                      batch_size: parseInt(e.target.value),
                    },
                  })
                }
                fullWidth
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Epochs"
                type="number"
                value={formData.config.epochs}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    config: {
                      ...formData.config,
                      epochs: parseInt(e.target.value),
                    },
                  })
                }
                fullWidth
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Optimizer</InputLabel>
                <Select
                  value={formData.config.optimizer}
                  label="Optimizer"
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      config: {
                        ...formData.config,
                        optimizer: e.target.value,
                      },
                    })
                  }
                >
                  <MenuItem value="adam">Adam</MenuItem>
                  <MenuItem value="sgd">SGD</MenuItem>
                  <MenuItem value="rmsprop">RMSprop</MenuItem>
                  <MenuItem value="adagrad">Adagrad</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Model Architecture</InputLabel>
                <Select
                  value={formData.config.architecture}
                  label="Model Architecture"
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      config: {
                        ...formData.config,
                        architecture: e.target.value,
                      },
                    })
                  }
                >
                  <MenuItem value="resnet50">ResNet-50</MenuItem>
                  <MenuItem value="vgg16">VGG-16</MenuItem>
                  <MenuItem value="mobilenet">MobileNet</MenuItem>
                  <MenuItem value="efficientnet">EfficientNet</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!formData.name}
        >
          {experiment ? 'Update' : 'Create'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

interface CompareDialogProps {
  open: boolean;
  onClose: () => void;
  experiments: Experiment[];
  comparison?: ExperimentComparison;
}

function CompareDialog({
  open,
  onClose,
  experiments,
  comparison,
}: CompareDialogProps) {
  if (!comparison) {
    return null;
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>Experiment Comparison</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 2 }}>
          {/* Learning Curves */}
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Learning Curves</Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={comparison.learning_curves}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <ChartTooltip />
                    <Legend />
                    {comparison.experiments.map((exp) => (
                      <Line
                        key={exp.id}
                        type="monotone"
                        dataKey={exp.name}
                        stroke={exp.color}
                        name={exp.name}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </Stack>
          </Paper>

          {/* Metrics Comparison */}
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Performance Metrics</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      {comparison.experiments.map((exp) => (
                        <TableCell key={exp.id} align="right">
                          {exp.name}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {comparison.metrics.map((metric) => (
                      <TableRow key={metric.name}>
                        <TableCell>{metric.name}</TableCell>
                        {metric.values.map((value, index) => (
                          <TableCell key={index} align="right">
                            {value.toFixed(4)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Stack>
          </Paper>

          {/* Hyperparameter Comparison */}
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Hyperparameters</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Parameter</TableCell>
                      {comparison.experiments.map((exp) => (
                        <TableCell key={exp.id} align="right">
                          {exp.name}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {comparison.hyperparameters.map((param) => (
                      <TableRow key={param.name}>
                        <TableCell>{param.name}</TableCell>
                        {param.values.map((value, index) => (
                          <TableCell key={index} align="right">
                            {value}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Stack>
          </Paper>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelExperiments() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const [experimentDialogOpen, setExperimentDialogOpen] = useState(false);
  const [compareDialogOpen, setCompareDialogOpen] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<
    Experiment | undefined
  >();
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const queryClient = useQueryClient();
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([
    'accuracy',
    'loss',
  ]);

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch experiments
  const { data: experiments, refetch: refetchExperiments } = useQuery({
    queryKey: ['model', modelId, 'experiments'],
    queryFn: () => modelApi.getExperiments(modelId!),
    enabled: !!modelId,
  });

  // Fetch experiment comparison
  const { data: comparison } = useQuery({
    queryKey: ['model', modelId, 'experiment-comparison', selectedExperiments],
    queryFn: () => modelApi.compareExperiments(modelId!, selectedExperiments),
    enabled: selectedExperiments.length >= 2,
  });

  // Create experiment mutation
  const { mutate: createExperiment } = useMutation({
    mutationFn: (data: any) => modelApi.createExperiment(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'experiments']);
      setMessage('Experiment created successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to create experiment');
      setSeverity('error');
    },
  });

  // Update experiment mutation
  const { mutate: updateExperiment } = useMutation({
    mutationFn: (data: any) =>
      modelApi.updateExperiment(modelId!, data.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'experiments']);
      setMessage('Experiment updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update experiment');
      setSeverity('error');
    },
  });

  // Delete experiment mutation
  const { mutate: deleteExperiment } = useMutation({
    mutationFn: (experimentId: string) =>
      modelApi.deleteExperiment(modelId!, experimentId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'experiments']);
      setMessage('Experiment deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete experiment');
      setSeverity('error');
    },
  });

  // Start experiment mutation
  const { mutate: startExperiment } = useMutation({
    mutationFn: (experimentId: string) =>
      modelApi.startExperiment(modelId!, experimentId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'experiments']);
      setMessage('Experiment started successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to start experiment');
      setSeverity('error');
    },
  });

  // Stop experiment mutation
  const { mutate: stopExperiment } = useMutation({
    mutationFn: (experimentId: string) =>
      modelApi.stopExperiment(modelId!, experimentId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'experiments']);
      setMessage('Experiment stopped successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to stop experiment');
      setSeverity('error');
    },
  });

  if (!model || !experiments) {
    return null;
  }

  const handleExperimentSelect = (experimentId: string) => {
    setSelectedExperiments((prev) => {
      if (prev.includes(experimentId)) {
        return prev.filter((id) => id !== experimentId);
      }
      return [...prev, experimentId];
    });
  };

  const handleCompare = () => {
    if (selectedExperiments.length >= 2) {
      setCompareDialogOpen(true);
    }
  };

  const handleDelete = (experimentId: string) => {
    if (
      confirm(
        'Are you sure you want to delete this experiment? This action cannot be undone.'
      )
    ) {
      deleteExperiment(experimentId);
    }
  };

  const handleSaveExperiment = (data: any) => {
    if (selectedExperiment) {
      updateExperiment({ ...data, id: selectedExperiment.id });
    } else {
      createExperiment(data);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'info';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'stopped':
        return 'warning';
      default:
        return 'default';
    }
  };

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
            Experiments
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage experiments for {model.name}
          </Typography>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => {
              setSelectedExperiment(undefined);
              setExperimentDialogOpen(true);
            }}
          >
            New Experiment
          </Button>
          <Button
            variant="outlined"
            startIcon={<CompareIcon />}
            onClick={handleCompare}
            disabled={selectedExperiments.length < 2}
          >
            Compare Selected
          </Button>
        </Stack>
      </Box>

      {/* Experiment List */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  indeterminate={
                    selectedExperiments.length > 0 &&
                    selectedExperiments.length < experiments.length
                  }
                  checked={
                    experiments.length > 0 &&
                    selectedExperiments.length === experiments.length
                  }
                  onChange={(e) =>
                    setSelectedExperiments(
                      e.target.checked
                        ? experiments.map((exp) => exp.id)
                        : []
                    )
                  }
                />
              </TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Metrics</TableCell>
              <TableCell>Runtime</TableCell>
              <TableCell>Created At</TableCell>
              <TableCell>Tags</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {experiments.map((experiment) => (
              <TableRow
                key={experiment.id}
                selected={selectedExperiments.includes(experiment.id)}
              >
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedExperiments.includes(experiment.id)}
                    onChange={() => handleExperimentSelect(experiment.id)}
                  />
                </TableCell>
                <TableCell>
                  <Stack direction="row" spacing={1} alignItems="center">
                    {experiment.is_best && (
                      <Tooltip title="Best Performing">
                        <StarIcon
                          fontSize="small"
                          sx={{ color: 'warning.main' }}
                        />
                      </Tooltip>
                    )}
                    <Stack spacing={0.5}>
                      <Typography variant="body2">{experiment.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {experiment.description}
                      </Typography>
                    </Stack>
                  </Stack>
                </TableCell>
                <TableCell>
                  <Chip
                    label={experiment.status}
                    color={getStatusColor(experiment.status)}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Stack spacing={0.5}>
                    <Typography variant="caption" color="text.secondary">
                      Accuracy:{' '}
                      {(experiment.metrics.accuracy * 100).toFixed(2)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Loss: {experiment.metrics.loss.toFixed(4)}
                    </Typography>
                  </Stack>
                </TableCell>
                <TableCell>
                  {experiment.runtime
                    ? `${Math.floor(experiment.runtime / 60)}m ${
                        experiment.runtime % 60
                      }s`
                    : '-'}
                </TableCell>
                <TableCell>
                  {format(
                    new Date(experiment.created_at),
                    'yyyy-MM-dd HH:mm:ss'
                  )}
                </TableCell>
                <TableCell>
                  <Stack direction="row" spacing={0.5}>
                    {experiment.tags.map((tag) => (
                      <Chip key={tag} label={tag} size="small" />
                    ))}
                  </Stack>
                </TableCell>
                <TableCell align="right">
                  <Stack direction="row" spacing={1} justifyContent="flex-end">
                    {experiment.status === 'running' ? (
                      <Tooltip title="Stop">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => stopExperiment(experiment.id)}
                        >
                          <StopIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    ) : (
                      <Tooltip title="Start">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => startExperiment(experiment.id)}
                          disabled={experiment.status === 'completed'}
                        >
                          <StartIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="Edit">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setSelectedExperiment(experiment);
                          setExperimentDialogOpen(true);
                        }}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDelete(experiment.id)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Experiment Dialog */}
      <ExperimentDialog
        open={experimentDialogOpen}
        onClose={() => {
          setExperimentDialogOpen(false);
          setSelectedExperiment(undefined);
        }}
        onSave={handleSaveExperiment}
        experiment={selectedExperiment}
      />

      {/* Compare Dialog */}
      <CompareDialog
        open={compareDialogOpen}
        onClose={() => {
          setCompareDialogOpen(false);
          setSelectedExperiments([]);
        }}
        experiments={experiments}
        comparison={comparison}
      />

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