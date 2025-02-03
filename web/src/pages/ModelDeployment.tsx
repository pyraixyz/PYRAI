import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Stack,
  Button,
  Paper,
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
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Sync as RollbackIcon,
  CloudUpload as DeployIcon,
  Settings as ConfigIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { modelApi } from '@/api/client';

interface Deployment {
  id: string;
  version: string;
  environment: 'development' | 'staging' | 'production';
  status: 'pending' | 'running' | 'stopped' | 'failed';
  resources: {
    cpu: string;
    memory: string;
    gpu: string;
  };
  scaling: {
    minReplicas: number;
    maxReplicas: number;
    targetCPUUtilization: number;
  };
  endpoint: string;
  created_at: string;
  updated_at: string;
  created_by: {
    id: string;
    name: string;
  };
}

interface DeploymentDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: any) => void;
  deployment?: Deployment;
}

function DeploymentDialog({
  open,
  onClose,
  onSave,
  deployment,
}: DeploymentDialogProps) {
  const [formData, setFormData] = useState(
    deployment || {
      version: '',
      environment: 'development',
      resources: {
        cpu: '1',
        memory: '2Gi',
        gpu: '0',
      },
      scaling: {
        minReplicas: 1,
        maxReplicas: 3,
        targetCPUUtilization: 80,
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
        {deployment ? 'Edit Deployment' : 'Create Deployment'}
      </DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <TextField
            label="Version"
            value={formData.version}
            onChange={(e) =>
              setFormData({ ...formData, version: e.target.value })
            }
            fullWidth
            required
          />

          <FormControl fullWidth>
            <InputLabel>Environment</InputLabel>
            <Select
              value={formData.environment}
              label="Environment"
              onChange={(e) =>
                setFormData({
                  ...formData,
                  environment: e.target.value as
                    | 'development'
                    | 'staging'
                    | 'production',
                })
              }
            >
              <MenuItem value="development">Development</MenuItem>
              <MenuItem value="staging">Staging</MenuItem>
              <MenuItem value="production">Production</MenuItem>
            </Select>
          </FormControl>

          <Typography variant="subtitle1" gutterBottom>
            Resources
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <TextField
                label="CPU"
                value={formData.resources.cpu}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    resources: { ...formData.resources, cpu: e.target.value },
                  })
                }
                fullWidth
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                label="Memory"
                value={formData.resources.memory}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    resources: { ...formData.resources, memory: e.target.value },
                  })
                }
                fullWidth
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                label="GPU"
                value={formData.resources.gpu}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    resources: { ...formData.resources, gpu: e.target.value },
                  })
                }
                fullWidth
              />
            </Grid>
          </Grid>

          <Typography variant="subtitle1" gutterBottom>
            Scaling
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <TextField
                label="Min Replicas"
                type="number"
                value={formData.scaling.minReplicas}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    scaling: {
                      ...formData.scaling,
                      minReplicas: parseInt(e.target.value),
                    },
                  })
                }
                fullWidth
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                label="Max Replicas"
                type="number"
                value={formData.scaling.maxReplicas}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    scaling: {
                      ...formData.scaling,
                      maxReplicas: parseInt(e.target.value),
                    },
                  })
                }
                fullWidth
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                label="Target CPU Utilization (%)"
                type="number"
                value={formData.scaling.targetCPUUtilization}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    scaling: {
                      ...formData.scaling,
                      targetCPUUtilization: parseInt(e.target.value),
                    },
                  })
                }
                fullWidth
              />
            </Grid>
          </Grid>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained">
          {deployment ? 'Update' : 'Deploy'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelDeployment() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [deploymentDialogOpen, setDeploymentDialogOpen] = useState(false);
  const [selectedDeployment, setSelectedDeployment] = useState<
    Deployment | undefined
  >();
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch deployments
  const { data: deployments, refetch: refetchDeployments } = useQuery({
    queryKey: ['model', modelId, 'deployments'],
    queryFn: () => modelApi.getDeployments(modelId!),
    enabled: !!modelId,
  });

  // Create deployment mutation
  const { mutate: createDeployment } = useMutation({
    mutationFn: (data: any) => modelApi.createDeployment(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'deployments']);
      setMessage('Deployment created successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to create deployment');
      setSeverity('error');
    },
  });

  // Update deployment mutation
  const { mutate: updateDeployment } = useMutation({
    mutationFn: (data: any) =>
      modelApi.updateDeployment(modelId!, data.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'deployments']);
      setMessage('Deployment updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update deployment');
      setSeverity('error');
    },
  });

  // Delete deployment mutation
  const { mutate: deleteDeployment } = useMutation({
    mutationFn: (deploymentId: string) =>
      modelApi.deleteDeployment(modelId!, deploymentId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'deployments']);
      setMessage('Deployment deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete deployment');
      setSeverity('error');
    },
  });

  // Start deployment mutation
  const { mutate: startDeployment } = useMutation({
    mutationFn: (deploymentId: string) =>
      modelApi.startDeployment(modelId!, deploymentId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'deployments']);
      setMessage('Deployment started successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to start deployment');
      setSeverity('error');
    },
  });

  // Stop deployment mutation
  const { mutate: stopDeployment } = useMutation({
    mutationFn: (deploymentId: string) =>
      modelApi.stopDeployment(modelId!, deploymentId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'deployments']);
      setMessage('Deployment stopped successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to stop deployment');
      setSeverity('error');
    },
  });

  if (!model || !deployments) {
    return null;
  }

  const handleSaveDeployment = (data: any) => {
    if (selectedDeployment) {
      updateDeployment({ ...data, id: selectedDeployment.id });
    } else {
      createDeployment(data);
    }
  };

  const handleDeleteDeployment = (deploymentId: string) => {
    if (confirm('Are you sure you want to delete this deployment?')) {
      deleteDeployment(deploymentId);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'pending':
        return 'warning';
      case 'stopped':
        return 'error';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getEnvironmentColor = (environment: string) => {
    switch (environment) {
      case 'production':
        return 'error';
      case 'staging':
        return 'warning';
      case 'development':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Deployments
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage deployments for {model.name}
        </Typography>
      </Box>

      {/* Actions */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        sx={{ mb: 3 }}
      >
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            setSelectedDeployment(undefined);
            setDeploymentDialogOpen(true);
          }}
        >
          Create Deployment
        </Button>

        <Tooltip title="Refresh">
          <IconButton onClick={() => refetchDeployments()}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Stack>

      {/* Deployments Table */}
      <TableContainer component={Paper} variant="outlined">
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Version</TableCell>
              <TableCell>Environment</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Resources</TableCell>
              <TableCell>Scaling</TableCell>
              <TableCell>Endpoint</TableCell>
              <TableCell>Created</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {deployments.map((deployment) => (
              <TableRow key={deployment.id}>
                <TableCell>{deployment.version}</TableCell>
                <TableCell>
                  <Chip
                    label={deployment.environment}
                    size="small"
                    color={getEnvironmentColor(deployment.environment)}
                  />
                </TableCell>
                <TableCell>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Chip
                      label={deployment.status}
                      size="small"
                      color={getStatusColor(deployment.status)}
                    />
                    {deployment.status === 'pending' && (
                      <LinearProgress sx={{ width: 50 }} />
                    )}
                  </Stack>
                </TableCell>
                <TableCell>
                  <Typography variant="caption" display="block">
                    CPU: {deployment.resources.cpu}
                  </Typography>
                  <Typography variant="caption" display="block">
                    Memory: {deployment.resources.memory}
                  </Typography>
                  <Typography variant="caption" display="block">
                    GPU: {deployment.resources.gpu}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="caption" display="block">
                    Min: {deployment.scaling.minReplicas}
                  </Typography>
                  <Typography variant="caption" display="block">
                    Max: {deployment.scaling.maxReplicas}
                  </Typography>
                  <Typography variant="caption" display="block">
                    CPU Target: {deployment.scaling.targetCPUUtilization}%
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography
                    variant="caption"
                    sx={{
                      maxWidth: 200,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      display: 'block',
                    }}
                  >
                    {deployment.endpoint}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="caption" display="block">
                    {format(new Date(deployment.created_at), 'yyyy-MM-dd HH:mm')}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    by {deployment.created_by.name}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Stack direction="row" spacing={1} justifyContent="flex-end">
                    {deployment.status === 'stopped' ? (
                      <Tooltip title="Start">
                        <IconButton
                          size="small"
                          onClick={() => startDeployment(deployment.id)}
                        >
                          <StartIcon />
                        </IconButton>
                      </Tooltip>
                    ) : (
                      <Tooltip title="Stop">
                        <IconButton
                          size="small"
                          onClick={() => stopDeployment(deployment.id)}
                        >
                          <StopIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="Edit">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setSelectedDeployment(deployment);
                          setDeploymentDialogOpen(true);
                        }}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        onClick={() => handleDeleteDeployment(deployment.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Deployment Dialog */}
      <DeploymentDialog
        open={deploymentDialogOpen}
        onClose={() => {
          setDeploymentDialogOpen(false);
          setSelectedDeployment(undefined);
        }}
        onSave={handleSaveDeployment}
        deployment={selectedDeployment}
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