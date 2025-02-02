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
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Autocomplete,
  Chip,
  Paper,
  Divider,
  Alert,
  Snackbar,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Help as HelpIcon,
  Settings as AdvancedIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi, datasetApi } from '@/api/client';

interface AdvancedSettingsDialogProps {
  open: boolean;
  onClose: () => void;
  config: Record<string, any>;
  onChange: (config: Record<string, any>) => void;
}

function AdvancedSettingsDialog({
  open,
  onClose,
  config,
  onChange,
}: AdvancedSettingsDialogProps) {
  const [localConfig, setLocalConfig] = useState(config);

  const handleChange = (key: string, value: any) => {
    setLocalConfig((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleSave = () => {
    onChange(localConfig);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Advanced Settings</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 2 }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Fine-tune your training process with advanced parameters
          </Typography>
          <Grid container spacing={2}>
            {/* Optimizer Settings */}
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Stack spacing={2}>
                  <Typography variant="subtitle1">Optimizer</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Type</InputLabel>
                        <Select
                          value={localConfig.optimizer_type}
                          label="Type"
                          onChange={(e) =>
                            handleChange('optimizer_type', e.target.value)
                          }
                        >
                          <MenuItem value="adam">Adam</MenuItem>
                          <MenuItem value="sgd">SGD</MenuItem>
                          <MenuItem value="adamw">AdamW</MenuItem>
                          <MenuItem value="rmsprop">RMSprop</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Learning Rate"
                        type="number"
                        value={localConfig.learning_rate}
                        onChange={(e) =>
                          handleChange('learning_rate', parseFloat(e.target.value))
                        }
                        inputProps={{ step: 0.0001, min: 0 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Weight Decay"
                        type="number"
                        value={localConfig.weight_decay}
                        onChange={(e) =>
                          handleChange('weight_decay', parseFloat(e.target.value))
                        }
                        inputProps={{ step: 0.0001, min: 0 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Momentum"
                        type="number"
                        value={localConfig.momentum}
                        onChange={(e) =>
                          handleChange('momentum', parseFloat(e.target.value))
                        }
                        inputProps={{ step: 0.1, min: 0, max: 1 }}
                        fullWidth
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </Paper>
            </Grid>

            {/* Scheduler Settings */}
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Stack spacing={2}>
                  <Typography variant="subtitle1">Learning Rate Scheduler</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Type</InputLabel>
                        <Select
                          value={localConfig.scheduler_type}
                          label="Type"
                          onChange={(e) =>
                            handleChange('scheduler_type', e.target.value)
                          }
                        >
                          <MenuItem value="step">StepLR</MenuItem>
                          <MenuItem value="cosine">CosineAnnealingLR</MenuItem>
                          <MenuItem value="reduce_on_plateau">
                            ReduceLROnPlateau
                          </MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Step Size"
                        type="number"
                        value={localConfig.scheduler_step_size}
                        onChange={(e) =>
                          handleChange(
                            'scheduler_step_size',
                            parseInt(e.target.value)
                          )
                        }
                        inputProps={{ min: 1 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Gamma"
                        type="number"
                        value={localConfig.scheduler_gamma}
                        onChange={(e) =>
                          handleChange(
                            'scheduler_gamma',
                            parseFloat(e.target.value)
                          )
                        }
                        inputProps={{ step: 0.1, min: 0, max: 1 }}
                        fullWidth
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </Paper>
            </Grid>

            {/* Regularization Settings */}
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Stack spacing={2}>
                  <Typography variant="subtitle1">Regularization</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={localConfig.use_dropout}
                            onChange={(e) =>
                              handleChange('use_dropout', e.target.checked)
                            }
                          />
                        }
                        label="Use Dropout"
                      />
                    </Grid>
                    {localConfig.use_dropout && (
                      <Grid item xs={12} sm={6}>
                        <TextField
                          label="Dropout Rate"
                          type="number"
                          value={localConfig.dropout_rate}
                          onChange={(e) =>
                            handleChange(
                              'dropout_rate',
                              parseFloat(e.target.value)
                            )
                          }
                          inputProps={{ step: 0.1, min: 0, max: 1 }}
                          fullWidth
                        />
                      </Grid>
                    )}
                  </Grid>
                </Stack>
              </Paper>
            </Grid>

            {/* Mixed Precision Settings */}
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Stack spacing={2}>
                  <Typography variant="subtitle1">Mixed Precision Training</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={localConfig.use_mixed_precision}
                            onChange={(e) =>
                              handleChange(
                                'use_mixed_precision',
                                e.target.checked
                              )
                            }
                          />
                        }
                        label="Enable Mixed Precision"
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </Paper>
            </Grid>
          </Grid>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleSave}>
          Save Changes
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelTraining() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const [advancedDialogOpen, setAdvancedDialogOpen] = useState(false);
  const queryClient = useQueryClient();

  // Basic training configuration
  const [config, setConfig] = useState({
    batch_size: 32,
    num_epochs: 100,
    validation_split: 0.2,
    dataset_id: '',
    optimizer_type: 'adam',
    learning_rate: 0.001,
    weight_decay: 0.0001,
    momentum: 0.9,
    scheduler_type: 'step',
    scheduler_step_size: 30,
    scheduler_gamma: 0.1,
    use_dropout: true,
    dropout_rate: 0.5,
    use_mixed_precision: true,
  });

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch available datasets
  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetApi.listDatasets({ page: 0, per_page: 100 }),
  });

  // Start training mutation
  const { mutate: startTraining, isLoading } = useMutation({
    mutationFn: (trainingConfig: any) =>
      modelApi.startTraining(modelId!, trainingConfig),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId]);
      setMessage('Training started successfully');
      setSeverity('success');
      // Navigate to model details page
      navigate(`/models/${modelId}`);
    },
    onError: () => {
      setMessage('Failed to start training');
      setSeverity('error');
    },
  });

  if (!model || !datasets) {
    return null;
  }

  const handleStartTraining = () => {
    startTraining(config);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Training Configuration
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Configure training parameters for {model.name}
        </Typography>
      </Box>

      {/* Content */}
      <Grid container spacing={3}>
        {/* Basic Settings */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Stack spacing={3}>
                <Typography variant="h6">Basic Settings</Typography>

                {/* Dataset Selection */}
                <FormControl fullWidth>
                  <Autocomplete
                    value={
                      datasets.items.find((d) => d.id === config.dataset_id) ||
                      null
                    }
                    onChange={(_, newValue) =>
                      setConfig((prev) => ({
                        ...prev,
                        dataset_id: newValue?.id || '',
                      }))
                    }
                    options={datasets.items}
                    getOptionLabel={(option) => option.name}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Training Dataset"
                        required
                      />
                    )}
                  />
                </FormControl>

                {/* Basic Training Parameters */}
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="Batch Size"
                      type="number"
                      value={config.batch_size}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          batch_size: parseInt(e.target.value),
                        }))
                      }
                      inputProps={{ min: 1 }}
                      fullWidth
                      required
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="Number of Epochs"
                      type="number"
                      value={config.num_epochs}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          num_epochs: parseInt(e.target.value),
                        }))
                      }
                      inputProps={{ min: 1 }}
                      fullWidth
                      required
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="Validation Split"
                      type="number"
                      value={config.validation_split}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          validation_split: parseFloat(e.target.value),
                        }))
                      }
                      inputProps={{ step: 0.1, min: 0, max: 1 }}
                      fullWidth
                      required
                    />
                  </Grid>
                </Grid>

                {/* Actions */}
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    mt: 2,
                  }}
                >
                  <Button
                    variant="outlined"
                    startIcon={<AdvancedIcon />}
                    onClick={() => setAdvancedDialogOpen(true)}
                  >
                    Advanced Settings
                  </Button>
                  <Stack direction="row" spacing={2}>
                    <Button
                      variant="outlined"
                      onClick={() => navigate(`/models/${modelId}`)}
                    >
                      Cancel
                    </Button>
                    <Button
                      variant="contained"
                      startIcon={<StartIcon />}
                      onClick={handleStartTraining}
                      disabled={isLoading || !config.dataset_id}
                    >
                      {isLoading ? 'Starting...' : 'Start Training'}
                    </Button>
                  </Stack>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Configuration Summary */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            <Card>
              <CardContent>
                <Stack spacing={2}>
                  <Typography variant="h6">Configuration Summary</Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText
                        primary="Model Architecture"
                        secondary={model.architecture}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Framework"
                        secondary={model.framework}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Dataset"
                        secondary={
                          datasets.items.find((d) => d.id === config.dataset_id)
                            ?.name || 'Not selected'
                        }
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Training Duration"
                        secondary={`${config.num_epochs} epochs`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Batch Size"
                        secondary={config.batch_size}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Validation Split"
                        secondary={`${(config.validation_split * 100).toFixed(
                          0
                        )}%`}
                      />
                    </ListItem>
                  </List>
                </Stack>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Stack spacing={2}>
                  <Typography variant="h6">Hardware Requirements</Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText
                        primary="Estimated Memory"
                        secondary="8 GB RAM"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="GPU Memory"
                        secondary="4 GB VRAM"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Storage"
                        secondary="2 GB free space"
                      />
                    </ListItem>
                  </List>
                </Stack>
              </CardContent>
            </Card>
          </Stack>
        </Grid>
      </Grid>

      {/* Advanced Settings Dialog */}
      <AdvancedSettingsDialog
        open={advancedDialogOpen}
        onClose={() => setAdvancedDialogOpen(false)}
        config={config}
        onChange={setConfig}
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