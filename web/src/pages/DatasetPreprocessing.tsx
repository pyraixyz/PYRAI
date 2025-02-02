import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  IconButton,
  Tooltip,
  Stack,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Alert,
  Snackbar,
  Paper,
  Chip,
  Divider,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  ArrowUpward as MoveUpIcon,
  ArrowDownward as MoveDownIcon,
  Preview as PreviewIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { datasetApi } from '@/api/client';

interface StepDialogProps {
  open: boolean;
  onClose: () => void;
  step?: PreprocessingStep;
  onSave: (step: PreprocessingStep) => void;
}

function StepDialog({ open, onClose, step, onSave }: StepDialogProps) {
  const [name, setName] = useState(step?.name || '');
  const [type, setType] = useState(step?.type || 'resize');
  const [params, setParams] = useState<Record<string, any>>(
    step?.params || getDefaultParams('resize')
  );

  const getDefaultParams = (type: string) => {
    switch (type) {
      case 'resize':
        return { width: 224, height: 224 };
      case 'normalize':
        return { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
      case 'augment':
        return {
          horizontal_flip: true,
          vertical_flip: false,
          rotation: 15,
          brightness: 0.2,
          contrast: 0.2,
        };
      case 'convert':
        return { format: 'RGB' };
      default:
        return {};
    }
  };

  const handleTypeChange = (newType: string) => {
    setType(newType);
    setParams(getDefaultParams(newType));
  };

  const handleParamChange = (key: string, value: any) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({
      id: step?.id || Date.now().toString(),
      name,
      type,
      params,
    });
    onClose();
  };

  const renderParamFields = () => {
    switch (type) {
      case 'resize':
        return (
          <>
            <TextField
              label="Width"
              type="number"
              value={params.width}
              onChange={(e) => handleParamChange('width', Number(e.target.value))}
              fullWidth
              required
            />
            <TextField
              label="Height"
              type="number"
              value={params.height}
              onChange={(e) => handleParamChange('height', Number(e.target.value))}
              fullWidth
              required
            />
          </>
        );
      case 'normalize':
        return (
          <>
            <TextField
              label="Mean (R,G,B)"
              value={params.mean.join(',')}
              onChange={(e) =>
                handleParamChange(
                  'mean',
                  e.target.value.split(',').map(Number)
                )
              }
              helperText="Comma-separated RGB values"
              fullWidth
              required
            />
            <TextField
              label="Std (R,G,B)"
              value={params.std.join(',')}
              onChange={(e) =>
                handleParamChange(
                  'std',
                  e.target.value.split(',').map(Number)
                )
              }
              helperText="Comma-separated RGB values"
              fullWidth
              required
            />
          </>
        );
      case 'augment':
        return (
          <>
            <FormControl fullWidth>
              <Stack spacing={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={params.horizontal_flip}
                      onChange={(e) =>
                        handleParamChange('horizontal_flip', e.target.checked)
                      }
                    />
                  }
                  label="Horizontal Flip"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={params.vertical_flip}
                      onChange={(e) =>
                        handleParamChange('vertical_flip', e.target.checked)
                      }
                    />
                  }
                  label="Vertical Flip"
                />
                <TextField
                  label="Rotation (degrees)"
                  type="number"
                  value={params.rotation}
                  onChange={(e) =>
                    handleParamChange('rotation', Number(e.target.value))
                  }
                  fullWidth
                />
                <TextField
                  label="Brightness Factor"
                  type="number"
                  value={params.brightness}
                  onChange={(e) =>
                    handleParamChange('brightness', Number(e.target.value))
                  }
                  inputProps={{ step: 0.1, min: 0, max: 1 }}
                  fullWidth
                />
                <TextField
                  label="Contrast Factor"
                  type="number"
                  value={params.contrast}
                  onChange={(e) =>
                    handleParamChange('contrast', Number(e.target.value))
                  }
                  inputProps={{ step: 0.1, min: 0, max: 1 }}
                  fullWidth
                />
              </Stack>
            </FormControl>
          </>
        );
      case 'convert':
        return (
          <FormControl fullWidth>
            <InputLabel>Format</InputLabel>
            <Select
              value={params.format}
              label="Format"
              onChange={(e) => handleParamChange('format', e.target.value)}
            >
              <MenuItem value="RGB">RGB</MenuItem>
              <MenuItem value="BGR">BGR</MenuItem>
              <MenuItem value="GRAY">Grayscale</MenuItem>
            </Select>
          </FormControl>
        );
      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>
          {step ? 'Edit Preprocessing Step' : 'Add Preprocessing Step'}
        </DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 2 }}>
            <TextField
              label="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              fullWidth
              required
            />
            <FormControl fullWidth>
              <InputLabel>Type</InputLabel>
              <Select
                value={type}
                label="Type"
                onChange={(e) => handleTypeChange(e.target.value)}
              >
                <MenuItem value="resize">Resize</MenuItem>
                <MenuItem value="normalize">Normalize</MenuItem>
                <MenuItem value="augment">Augment</MenuItem>
                <MenuItem value="convert">Convert</MenuItem>
              </Select>
            </FormControl>
            {renderParamFields()}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            disabled={!name}
          >
            Save
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
}

interface PreviewDialogProps {
  open: boolean;
  onClose: () => void;
  datasetId: string;
  fileId: string;
  stepId?: string;
}

function PreviewDialog({
  open,
  onClose,
  datasetId,
  fileId,
  stepId,
}: PreviewDialogProps) {
  const { data: preview } = useQuery({
    queryKey: ['preview', datasetId, fileId, stepId],
    queryFn: () => datasetApi.previewPreprocessing(datasetId, fileId, stepId),
    enabled: open,
  });

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Preview</DialogTitle>
      <DialogContent>
        <Stack spacing={2}>
          {preview ? (
            <>
              <Box
                component="img"
                src={preview.image_url}
                alt="Preview"
                sx={{
                  width: '100%',
                  height: 'auto',
                  maxHeight: '60vh',
                  objectFit: 'contain',
                }}
              />
              {preview.metadata && (
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Metadata
                  </Typography>
                  <Grid container spacing={2}>
                    {Object.entries(preview.metadata).map(([key, value]) => (
                      <Grid item xs={6} key={key}>
                        <Typography variant="caption" color="text.secondary">
                          {key}
                        </Typography>
                        <Typography variant="body2">
                          {JSON.stringify(value)}
                        </Typography>
                      </Grid>
                    ))}
                  </Grid>
                </Paper>
              )}
            </>
          ) : (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: 300,
              }}
            >
              <CircularProgress />
            </Box>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default function DatasetPreprocessing() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [stepDialogOpen, setStepDialogOpen] = useState(false);
  const [selectedStep, setSelectedStep] = useState<PreprocessingStep | null>(null);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const queryClient = useQueryClient();

  // Fetch dataset
  const { data: dataset } = useQuery({
    queryKey: ['dataset', datasetId],
    queryFn: () => datasetApi.getDataset(datasetId!),
    enabled: !!datasetId,
  });

  // Fetch preprocessing pipeline
  const { data: pipeline } = useQuery({
    queryKey: ['pipeline', datasetId],
    queryFn: () => datasetApi.getPreprocessingPipeline(datasetId!),
    enabled: !!datasetId,
  });

  // Update pipeline mutation
  const { mutate: updatePipeline } = useMutation({
    mutationFn: (steps: PreprocessingStep[]) =>
      datasetApi.updatePreprocessingPipeline(datasetId!, steps),
    onSuccess: () => {
      queryClient.invalidateQueries(['pipeline', datasetId]);
      setMessage('Pipeline updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update pipeline');
      setSeverity('error');
    },
  });

  // Start preprocessing mutation
  const { mutate: startPreprocessing } = useMutation({
    mutationFn: () => datasetApi.startPreprocessing(datasetId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['dataset', datasetId]);
      setMessage('Preprocessing started');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to start preprocessing');
      setSeverity('error');
    },
  });

  // Stop preprocessing mutation
  const { mutate: stopPreprocessing } = useMutation({
    mutationFn: () => datasetApi.stopPreprocessing(datasetId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['dataset', datasetId]);
      setMessage('Preprocessing stopped');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to stop preprocessing');
      setSeverity('error');
    },
  });

  if (!dataset || !pipeline) {
    return null;
  }

  const handleSaveStep = (step: PreprocessingStep) => {
    const newSteps = selectedStep
      ? pipeline.steps.map((s) => (s.id === step.id ? step : s))
      : [...pipeline.steps, step];
    updatePipeline(newSteps);
    setSelectedStep(null);
  };

  const handleDeleteStep = (stepId: string) => {
    if (confirm('Are you sure you want to delete this step?')) {
      updatePipeline(pipeline.steps.filter((s) => s.id !== stepId));
    }
  };

  const handleMoveStep = (stepId: string, direction: 'up' | 'down') => {
    const index = pipeline.steps.findIndex((s) => s.id === stepId);
    if (
      (direction === 'up' && index === 0) ||
      (direction === 'down' && index === pipeline.steps.length - 1)
    ) {
      return;
    }

    const newSteps = [...pipeline.steps];
    const newIndex = direction === 'up' ? index - 1 : index + 1;
    [newSteps[index], newSteps[newIndex]] = [
      newSteps[newIndex],
      newSteps[index],
    ];
    updatePipeline(newSteps);
  };

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Box>
          <Typography variant="h5" gutterBottom>
            Preprocessing Pipeline
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {dataset.name}
          </Typography>
        </Box>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={() => {
              setSelectedStep(null);
              setStepDialogOpen(true);
            }}
          >
            Add Step
          </Button>
          {dataset.is_preprocessing ? (
            <Button
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
              onClick={() => stopPreprocessing()}
            >
              Stop
            </Button>
          ) : (
            <Button
              variant="contained"
              startIcon={<StartIcon />}
              onClick={() => startPreprocessing()}
              disabled={pipeline.steps.length === 0}
            >
              Start
            </Button>
          )}
        </Stack>
      </Box>

      {/* Progress */}
      {dataset.is_preprocessing && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Stack spacing={2}>
              <Stack
                direction="row"
                spacing={2}
                alignItems="center"
                justifyContent="space-between"
              >
                <Typography variant="subtitle1">Processing...</Typography>
                <Typography variant="body2" color="text.secondary">
                  {Math.round(dataset.processed_percentage * 100)}% Complete
                </Typography>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={dataset.processed_percentage * 100}
              />
              <Typography variant="caption" color="text.secondary">
                Started at:{' '}
                {format(
                  new Date(dataset.preprocessing_started_at!),
                  'yyyy-MM-dd HH:mm:ss'
                )}
              </Typography>
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* Pipeline Steps */}
      <Card>
        <CardContent>
          <Stepper orientation="vertical">
            {pipeline.steps.map((step, index) => (
              <Step key={step.id} active={true}>
                <StepLabel>
                  <Stack
                    direction="row"
                    spacing={1}
                    alignItems="center"
                    justifyContent="space-between"
                    sx={{ width: '100%' }}
                  >
                    <Typography>{step.name}</Typography>
                    <Stack direction="row" spacing={1}>
                      <Tooltip title="Preview">
                        <IconButton
                          size="small"
                          onClick={() => {
                            setSelectedFile(dataset.preview_file_id);
                            setSelectedStep(step);
                            setPreviewDialogOpen(true);
                          }}
                        >
                          <PreviewIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit">
                        <IconButton
                          size="small"
                          onClick={() => {
                            setSelectedStep(step);
                            setStepDialogOpen(true);
                          }}
                        >
                          <EditIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Move Up">
                        <IconButton
                          size="small"
                          onClick={() => handleMoveStep(step.id, 'up')}
                          disabled={index === 0}
                        >
                          <MoveUpIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Move Down">
                        <IconButton
                          size="small"
                          onClick={() => handleMoveStep(step.id, 'down')}
                          disabled={index === pipeline.steps.length - 1}
                        >
                          <MoveDownIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDeleteStep(step.id)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                  </Stack>
                </StepLabel>
                <StepContent>
                  <Box sx={{ ml: 2 }}>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      gutterBottom
                    >
                      Type: {step.type}
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      {Object.entries(step.params).map(([key, value]) => (
                        <Chip
                          key={key}
                          label={`${key}: ${JSON.stringify(value)}`}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Stack>
                  </Box>
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>

      {/* Dialogs */}
      <StepDialog
        open={stepDialogOpen}
        onClose={() => {
          setStepDialogOpen(false);
          setSelectedStep(null);
        }}
        step={selectedStep || undefined}
        onSave={handleSaveStep}
      />
      {selectedFile && (
        <PreviewDialog
          open={previewDialogOpen}
          onClose={() => {
            setPreviewDialogOpen(false);
            setSelectedFile(null);
            setSelectedStep(null);
          }}
          datasetId={datasetId!}
          fileId={selectedFile}
          stepId={selectedStep?.id}
        />
      )}

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