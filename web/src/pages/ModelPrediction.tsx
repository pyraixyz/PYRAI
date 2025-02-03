import { useState, useCallback } from 'react';
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Tabs,
  Tab,
} from '@mui/material';
import {
  PlayArrow as PredictIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  History as HistoryIcon,
  CloudUpload as BatchIcon,
  Code as APIIcon,
  Settings as ConfigIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { useDropzone } from 'react-dropzone';
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
      id={`prediction-tabpanel-${index}`}
      aria-labelledby={`prediction-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface BatchPredictionProps {
  modelId: string;
  onPredict: (file: File) => void;
  isLoading: boolean;
}

function BatchPrediction({ modelId, onPredict, isLoading }: BatchPredictionProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onPredict(acceptedFiles[0]);
      }
    },
    [onPredict]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
  });

  return (
    <Stack spacing={3}>
      {/* Upload Area */}
      <Paper
        variant="outlined"
        sx={{
          p: 3,
          textAlign: 'center',
          backgroundColor: (theme) =>
            isDragActive ? theme.palette.action.hover : 'inherit',
          cursor: 'pointer',
        }}
        {...getRootProps()}
      >
        <input {...getInputProps()} />
        <Stack spacing={2} alignItems="center">
          <BatchIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
          {isLoading ? (
            <CircularProgress />
          ) : (
            <>
              <Typography variant="h6">
                {isDragActive
                  ? 'Drop the file here'
                  : 'Drag and drop your file here'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Support CSV and JSON formats
              </Typography>
              <Button variant="outlined" startIcon={<UploadIcon />}>
                Select File
              </Button>
            </>
          )}
        </Stack>
      </Paper>

      {/* Instructions */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Instructions</Typography>
          <List dense>
            <ListItem>
              <ListItemText
                primary="File Format"
                secondary="CSV file with headers or JSON array of objects"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Maximum Size"
                secondary="100MB per file"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Processing Time"
                secondary="Depends on file size and model complexity"
              />
            </ListItem>
          </List>
        </Stack>
      </Paper>
    </Stack>
  );
}

interface SinglePredictionProps {
  modelId: string;
  onPredict: (data: any) => void;
  isLoading: boolean;
}

function SinglePrediction({
  modelId,
  onPredict,
  isLoading,
}: SinglePredictionProps) {
  const [input, setInput] = useState<any>({});

  // Fetch model input schema
  const { data: schema } = useQuery({
    queryKey: ['model', modelId, 'schema'],
    queryFn: () => modelApi.getInputSchema(modelId),
  });

  const handlePredict = () => {
    onPredict(input);
  };

  if (!schema) {
    return null;
  }

  return (
    <Stack spacing={3}>
      {/* Input Form */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={3}>
          <Typography variant="subtitle1">Input Parameters</Typography>
          <Grid container spacing={2}>
            {schema.properties.map((prop) => (
              <Grid item xs={12} sm={6} key={prop.name}>
                {prop.type === 'number' ? (
                  <TextField
                    label={prop.label}
                    type="number"
                    value={input[prop.name] || ''}
                    onChange={(e) =>
                      setInput((prev) => ({
                        ...prev,
                        [prop.name]: parseFloat(e.target.value),
                      }))
                    }
                    helperText={prop.description}
                    fullWidth
                  />
                ) : prop.type === 'select' ? (
                  <FormControl fullWidth>
                    <InputLabel>{prop.label}</InputLabel>
                    <Select
                      value={input[prop.name] || ''}
                      label={prop.label}
                      onChange={(e) =>
                        setInput((prev) => ({
                          ...prev,
                          [prop.name]: e.target.value,
                        }))
                      }
                    >
                      {prop.options.map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                          {option.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                ) : (
                  <TextField
                    label={prop.label}
                    value={input[prop.name] || ''}
                    onChange={(e) =>
                      setInput((prev) => ({
                        ...prev,
                        [prop.name]: e.target.value,
                      }))
                    }
                    helperText={prop.description}
                    fullWidth
                  />
                )}
              </Grid>
            ))}
          </Grid>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="contained"
              startIcon={<PredictIcon />}
              onClick={handlePredict}
              disabled={isLoading}
            >
              {isLoading ? 'Predicting...' : 'Predict'}
            </Button>
          </Box>
        </Stack>
      </Paper>

      {/* Example */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Example Input</Typography>
          <Typography
            variant="body2"
            component="pre"
            sx={{
              p: 2,
              backgroundColor: 'background.default',
              borderRadius: 1,
              overflow: 'auto',
            }}
          >
            {JSON.stringify(schema.example, null, 2)}
          </Typography>
        </Stack>
      </Paper>
    </Stack>
  );
}

interface PredictionHistoryProps {
  modelId: string;
  onDownload: (predictionId: string) => void;
  onDelete: (predictionId: string) => void;
}

function PredictionHistory({
  modelId,
  onDownload,
  onDelete,
}: PredictionHistoryProps) {
  // Fetch prediction history
  const { data: history } = useQuery({
    queryKey: ['model', modelId, 'predictions'],
    queryFn: () => modelApi.listPredictions(modelId),
  });

  if (!history) {
    return null;
  }

  return (
    <TableContainer component={Paper} variant="outlined">
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>ID</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Created At</TableCell>
            <TableCell>Duration</TableCell>
            <TableCell>Size</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {history.map((prediction) => (
            <TableRow key={prediction.id}>
              <TableCell>{prediction.id}</TableCell>
              <TableCell>
                <Chip
                  label={prediction.type}
                  size="small"
                  color={prediction.type === 'batch' ? 'primary' : 'default'}
                />
              </TableCell>
              <TableCell>
                <Chip
                  label={prediction.status}
                  size="small"
                  color={
                    prediction.status === 'completed'
                      ? 'success'
                      : prediction.status === 'failed'
                      ? 'error'
                      : 'warning'
                  }
                />
              </TableCell>
              <TableCell>
                {format(new Date(prediction.created_at), 'yyyy-MM-dd HH:mm:ss')}
              </TableCell>
              <TableCell>
                {prediction.duration
                  ? `${(prediction.duration / 1000).toFixed(2)}s`
                  : '-'}
              </TableCell>
              <TableCell>
                {prediction.size ? `${prediction.size} rows` : '-'}
              </TableCell>
              <TableCell align="right">
                <Stack direction="row" spacing={1} justifyContent="flex-end">
                  <Tooltip title="Download Results">
                    <IconButton
                      size="small"
                      onClick={() => onDownload(prediction.id)}
                      disabled={prediction.status !== 'completed'}
                    >
                      <DownloadIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Delete">
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => onDelete(prediction.id)}
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
  );
}

export default function ModelPrediction() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const [tabValue, setTabValue] = useState(0);
  const queryClient = useQueryClient();

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Single prediction mutation
  const { mutate: predictSingle, isLoading: isSingleLoading } = useMutation({
    mutationFn: (data: any) => modelApi.predict(modelId!, data),
    onSuccess: (result) => {
      queryClient.invalidateQueries(['model', modelId, 'predictions']);
      setMessage('Prediction completed successfully');
      setSeverity('success');
      // Handle prediction result
    },
    onError: () => {
      setMessage('Failed to make prediction');
      setSeverity('error');
    },
  });

  // Batch prediction mutation
  const { mutate: predictBatch, isLoading: isBatchLoading } = useMutation({
    mutationFn: (file: File) => modelApi.predictBatch(modelId!, file),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'predictions']);
      setMessage('Batch prediction started successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to start batch prediction');
      setSeverity('error');
    },
  });

  // Download prediction mutation
  const { mutate: downloadPrediction } = useMutation({
    mutationFn: (predictionId: string) =>
      modelApi.downloadPrediction(modelId!, predictionId),
    onSuccess: () => {
      setMessage('Results downloaded successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to download results');
      setSeverity('error');
    },
  });

  // Delete prediction mutation
  const { mutate: deletePrediction } = useMutation({
    mutationFn: (predictionId: string) =>
      modelApi.deletePrediction(modelId!, predictionId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'predictions']);
      setMessage('Prediction deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete prediction');
      setSeverity('error');
    },
  });

  if (!model) {
    return null;
  }

  const handleDelete = (predictionId: string) => {
    if (
      confirm(
        'Are you sure you want to delete this prediction? This action cannot be undone.'
      )
    ) {
      deletePrediction(predictionId);
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Prediction
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Make predictions with {model.name}
        </Typography>
      </Box>

      {/* Content */}
      <Card>
        <CardContent>
          <Tabs
            value={tabValue}
            onChange={(_, newValue) => setTabValue(newValue)}
            aria-label="prediction tabs"
          >
            <Tab
              icon={<PredictIcon />}
              label="Single Prediction"
              id="prediction-tab-0"
              aria-controls="prediction-tabpanel-0"
            />
            <Tab
              icon={<BatchIcon />}
              label="Batch Prediction"
              id="prediction-tab-1"
              aria-controls="prediction-tabpanel-1"
            />
            <Tab
              icon={<HistoryIcon />}
              label="History"
              id="prediction-tab-2"
              aria-controls="prediction-tabpanel-2"
            />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <SinglePrediction
              modelId={modelId!}
              onPredict={predictSingle}
              isLoading={isSingleLoading}
            />
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <BatchPrediction
              modelId={modelId!}
              onPredict={predictBatch}
              isLoading={isBatchLoading}
            />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <PredictionHistory
              modelId={modelId!}
              onDownload={downloadPrediction}
              onDelete={handleDelete}
            />
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