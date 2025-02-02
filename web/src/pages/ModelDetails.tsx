import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Tabs,
  Tab,
  Stack,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  LinearProgress,
  Alert,
  Snackbar,
  Paper,
  List,
  ListItem,
  ListItemText,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Download as ExportIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Assessment as EvalIcon,
  Timeline as MetricsIcon,
  Code as CodeIcon,
  Save as SaveIcon,
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
      id={`model-tabpanel-${index}`}
      aria-labelledby={`model-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface EditModelDialogProps {
  open: boolean;
  onClose: () => void;
  model: Model;
}

function EditModelDialog({ open, onClose, model }: EditModelDialogProps) {
  const [name, setName] = useState(model.name);
  const [description, setDescription] = useState(model.description || '');
  const queryClient = useQueryClient();

  const { mutate: updateModel, isLoading } = useMutation({
    mutationFn: (data: Partial<Model>) => modelApi.updateModel(model.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', model.id]);
      onClose();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateModel({ name, description });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>Edit Model</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 2 }}>
            <TextField
              label="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              fullWidth
              required
            />
            <TextField
              label="Description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              multiline
              rows={4}
              fullWidth
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            disabled={isLoading || !name}
          >
            {isLoading ? 'Saving...' : 'Save'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
}

function Overview({ model }: { model: Model }) {
  return (
    <Stack spacing={3}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle2" color="text.secondary">
                Architecture
              </Typography>
              <Typography variant="body1">{model.architecture}</Typography>
              <Typography variant="subtitle2" color="text.secondary">
                Framework
              </Typography>
              <Typography variant="body1">{model.framework}</Typography>
              <Typography variant="subtitle2" color="text.secondary">
                Created At
              </Typography>
              <Typography variant="body1">
                {format(new Date(model.created_at), 'yyyy-MM-dd HH:mm:ss')}
              </Typography>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle2" color="text.secondary">
                Status
              </Typography>
              <Chip
                label={model.status}
                color={
                  model.status === 'ready'
                    ? 'success'
                    : model.status === 'training'
                    ? 'warning'
                    : 'default'
                }
              />
              <Typography variant="subtitle2" color="text.secondary">
                Size
              </Typography>
              <Typography variant="body1">
                {(model.size / (1024 * 1024)).toFixed(2)} MB
              </Typography>
              <Typography variant="subtitle2" color="text.secondary">
                Last Updated
              </Typography>
              <Typography variant="body1">
                {format(new Date(model.updated_at), 'yyyy-MM-dd HH:mm:ss')}
              </Typography>
            </Stack>
          </Paper>
        </Grid>
      </Grid>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle2" color="text.secondary">
            Description
          </Typography>
          <Typography variant="body1">
            {model.description || 'No description provided.'}
          </Typography>
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle2" color="text.secondary">
            Training Configuration
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableBody>
                {Object.entries(model.training_config || {}).map(([key, value]) => (
                  <TableRow key={key}>
                    <TableCell
                      component="th"
                      scope="row"
                      sx={{ width: '30%', color: 'text.secondary' }}
                    >
                      {key}
                    </TableCell>
                    <TableCell>{JSON.stringify(value)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Stack>
      </Paper>
    </Stack>
  );
}

function TrainingHistory({ model }: { model: Model }) {
  const { data: history } = useQuery({
    queryKey: ['model', model.id, 'history'],
    queryFn: () => modelApi.getTrainingHistory(model.id),
  });

  if (!history) {
    return null;
  }

  return (
    <Stack spacing={3}>
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Training Progress</Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history.metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <ChartTooltip />
                    <Legend />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="loss"
                      stroke="#8884d8"
                      name="Loss"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#82ca9d"
                      name="Accuracy"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} lg={4}>
          <Stack spacing={2}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Stack spacing={2}>
                <Typography variant="subtitle2" color="text.secondary">
                  Current Epoch
                </Typography>
                <Typography variant="h4">
                  {history.current_epoch} / {history.total_epochs}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={(history.current_epoch / history.total_epochs) * 100}
                />
              </Stack>
            </Paper>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Stack spacing={2}>
                <Typography variant="subtitle2" color="text.secondary">
                  Best Metrics
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText
                      primary="Loss"
                      secondary={history.best_metrics.loss.toFixed(4)}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Accuracy"
                      secondary={`${(history.best_metrics.accuracy * 100).toFixed(
                        2
                      )}%`}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Epoch"
                      secondary={history.best_metrics.epoch}
                    />
                  </ListItem>
                </List>
              </Stack>
            </Paper>
          </Stack>
        </Grid>
      </Grid>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Training Log</Typography>
          <Box
            sx={{
              maxHeight: 300,
              overflow: 'auto',
              bgcolor: 'grey.50',
              p: 2,
              borderRadius: 1,
              fontFamily: 'monospace',
              fontSize: '0.875rem',
            }}
          >
            {history.logs.map((log, index) => (
              <Box key={index} sx={{ mb: 1 }}>
                <Typography
                  variant="caption"
                  component="span"
                  sx={{ color: 'text.secondary', mr: 2 }}
                >
                  {format(new Date(log.timestamp), 'HH:mm:ss')}
                </Typography>
                <Typography component="span">{log.message}</Typography>
              </Box>
            ))}
          </Box>
        </Stack>
      </Paper>
    </Stack>
  );
}

function Evaluation({ model }: { model: Model }) {
  const { data: evaluation } = useQuery({
    queryKey: ['model', model.id, 'evaluation'],
    queryFn: () => modelApi.getEvaluation(model.id),
  });

  if (!evaluation) {
    return null;
  }

  return (
    <Stack spacing={3}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Performance Metrics</Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    {Object.entries(evaluation.metrics).map(([key, value]) => (
                      <TableRow key={key}>
                        <TableCell
                          component="th"
                          scope="row"
                          sx={{ color: 'text.secondary' }}
                        >
                          {key}
                        </TableCell>
                        <TableCell align="right">
                          {typeof value === 'number'
                            ? value.toFixed(4)
                            : JSON.stringify(value)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Resource Usage</Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Memory Usage"
                    secondary={`${(
                      evaluation.resource_usage.memory / (1024 * 1024 * 1024)
                    ).toFixed(2)} GB`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="GPU Memory"
                    secondary={`${(
                      evaluation.resource_usage.gpu_memory / (1024 * 1024 * 1024)
                    ).toFixed(2)} GB`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Inference Time"
                    secondary={`${evaluation.resource_usage.inference_time.toFixed(
                      2
                    )} ms`}
                  />
                </ListItem>
              </List>
            </Stack>
          </Paper>
        </Grid>
      </Grid>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Test Results</Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Dataset</TableCell>
                  <TableCell align="right">Accuracy</TableCell>
                  <TableCell align="right">Loss</TableCell>
                  <TableCell align="right">F1 Score</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {evaluation.test_results.map((result) => (
                  <TableRow key={result.dataset}>
                    <TableCell>{result.dataset}</TableCell>
                    <TableCell align="right">
                      {(result.accuracy * 100).toFixed(2)}%
                    </TableCell>
                    <TableCell align="right">{result.loss.toFixed(4)}</TableCell>
                    <TableCell align="right">
                      {result.f1_score.toFixed(4)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Stack>
      </Paper>
    </Stack>
  );
}

export default function ModelDetails() {
  const { modelId } = useParams<{ modelId: string }>();
  const [tabValue, setTabValue] = useState(0);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const queryClient = useQueryClient();

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Start training mutation
  const { mutate: startTraining } = useMutation({
    mutationFn: () => modelApi.startTraining(modelId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId]);
      setMessage('Training started successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to start training');
      setSeverity('error');
    },
  });

  // Stop training mutation
  const { mutate: stopTraining } = useMutation({
    mutationFn: () => modelApi.stopTraining(modelId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId]);
      setMessage('Training stopped successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to stop training');
      setSeverity('error');
    },
  });

  // Export model mutation
  const { mutate: exportModel } = useMutation({
    mutationFn: () => modelApi.exportModel(modelId!),
    onSuccess: () => {
      setMessage('Model exported successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to export model');
      setSeverity('error');
    },
  });

  // Delete model mutation
  const { mutate: deleteModel } = useMutation({
    mutationFn: () => modelApi.deleteModel(modelId!),
    onSuccess: () => {
      // Navigate to models list
      window.location.href = '/models';
    },
    onError: () => {
      setMessage('Failed to delete model');
      setSeverity('error');
    },
  });

  if (!model) {
    return null;
  }

  const handleDelete = () => {
    if (
      confirm(
        'Are you sure you want to delete this model? This action cannot be undone.'
      )
    ) {
      deleteModel();
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
            {model.name}
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip
              label={model.status}
              color={
                model.status === 'ready'
                  ? 'success'
                  : model.status === 'training'
                  ? 'warning'
                  : 'default'
              }
              size="small"
            />
            <Typography variant="body2" color="text.secondary">
              Last updated {format(new Date(model.updated_at), 'PPp')}
            </Typography>
          </Stack>
        </Box>
        <Stack direction="row" spacing={1}>
          {model.status === 'training' ? (
            <Button
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
              onClick={() => stopTraining()}
            >
              Stop Training
            </Button>
          ) : (
            <Button
              variant="contained"
              startIcon={<StartIcon />}
              onClick={() => startTraining()}
            >
              Start Training
            </Button>
          )}
          <Button
            variant="outlined"
            startIcon={<ExportIcon />}
            onClick={() => exportModel()}
          >
            Export
          </Button>
          <Tooltip title="Edit">
            <IconButton
              onClick={() => setEditDialogOpen(true)}
              sx={{ bgcolor: 'background.paper' }}
            >
              <EditIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton
              onClick={handleDelete}
              color="error"
              sx={{ bgcolor: 'background.paper' }}
            >
              <DeleteIcon />
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
            aria-label="model details tabs"
          >
            <Tab
              icon={<CodeIcon />}
              label="Overview"
              id="model-tab-0"
              aria-controls="model-tabpanel-0"
            />
            <Tab
              icon={<MetricsIcon />}
              label="Training History"
              id="model-tab-1"
              aria-controls="model-tabpanel-1"
            />
            <Tab
              icon={<EvalIcon />}
              label="Evaluation"
              id="model-tab-2"
              aria-controls="model-tabpanel-2"
            />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <Overview model={model} />
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <TrainingHistory model={model} />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <Evaluation model={model} />
          </TabPanel>
        </CardContent>
      </Card>

      {/* Dialogs */}
      <EditModelDialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        model={model}
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