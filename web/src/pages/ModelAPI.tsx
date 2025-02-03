import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Stack,
  Button,
  Paper,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Snackbar,
  Divider,
} from '@mui/material';
import {
  PlayArrow as TestIcon,
  Code as CodeIcon,
  Key as KeyIcon,
  History as HistoryIcon,
  ContentCopy as CopyIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Download as DownloadIcon,
  Security as SecurityIcon,
  Speed as PerformanceIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { modelApi } from '@/api/client';

interface APIEndpoint {
  id: string;
  name: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  parameters: {
    name: string;
    type: string;
    required: boolean;
    description: string;
    default?: any;
  }[];
  responses: {
    code: number;
    description: string;
    schema: any;
  }[];
  authentication: boolean;
  rate_limit: number;
  created_at: string;
  updated_at: string;
}

interface APIKey {
  id: string;
  name: string;
  key: string;
  created_at: string;
  expires_at: string | null;
  last_used_at: string | null;
  status: 'active' | 'expired' | 'revoked';
}

interface APITest {
  endpoint_id: string;
  parameters: Record<string, any>;
  headers: Record<string, string>;
  response?: {
    status: number;
    headers: Record<string, string>;
    body: any;
    latency: number;
  };
  created_at: string;
}

interface EndpointDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: Partial<APIEndpoint>) => void;
  endpoint?: APIEndpoint;
}

interface APIKeyDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: { name: string; expires_at?: string }) => void;
}

function EndpointDialog({
  open,
  onClose,
  onSave,
  endpoint,
}: EndpointDialogProps) {
  const [formData, setFormData] = useState(
    endpoint || {
      name: '',
      method: 'POST',
      path: '',
      description: '',
      parameters: [],
      responses: [],
      authentication: true,
      rate_limit: 60,
    }
  );

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {endpoint ? 'Edit Endpoint' : 'Create Endpoint'}
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

          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Method</InputLabel>
                <Select
                  value={formData.method}
                  label="Method"
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      method: e.target.value as any,
                    })
                  }
                >
                  <MenuItem value="GET">GET</MenuItem>
                  <MenuItem value="POST">POST</MenuItem>
                  <MenuItem value="PUT">PUT</MenuItem>
                  <MenuItem value="DELETE">DELETE</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={8}>
              <TextField
                label="Path"
                value={formData.path}
                onChange={(e) =>
                  setFormData({ ...formData, path: e.target.value })
                }
                fullWidth
                required
              />
            </Grid>
          </Grid>

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

          <FormControl fullWidth>
            <InputLabel>Rate Limit (requests per minute)</InputLabel>
            <Select
              value={formData.rate_limit}
              label="Rate Limit (requests per minute)"
              onChange={(e) =>
                setFormData({
                  ...formData,
                  rate_limit: e.target.value as number,
                })
              }
            >
              <MenuItem value={60}>60</MenuItem>
              <MenuItem value={300}>300</MenuItem>
              <MenuItem value={1000}>1000</MenuItem>
              <MenuItem value={3000}>3000</MenuItem>
            </Select>
          </FormControl>

          {/* Parameters Section */}
          <Typography variant="subtitle1" gutterBottom>
            Parameters
          </Typography>
          {formData.parameters.map((param, index) => (
            <Grid container spacing={2} key={index}>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Name"
                  value={param.name}
                  onChange={(e) => {
                    const newParams = [...formData.parameters];
                    newParams[index].name = e.target.value;
                    setFormData({ ...formData, parameters: newParams });
                  }}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Type</InputLabel>
                  <Select
                    value={param.type}
                    label="Type"
                    onChange={(e) => {
                      const newParams = [...formData.parameters];
                      newParams[index].type = e.target.value;
                      setFormData({ ...formData, parameters: newParams });
                    }}
                  >
                    <MenuItem value="string">String</MenuItem>
                    <MenuItem value="number">Number</MenuItem>
                    <MenuItem value="boolean">Boolean</MenuItem>
                    <MenuItem value="object">Object</MenuItem>
                    <MenuItem value="array">Array</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={5}>
                <TextField
                  label="Description"
                  value={param.description}
                  onChange={(e) => {
                    const newParams = [...formData.parameters];
                    newParams[index].description = e.target.value;
                    setFormData({ ...formData, parameters: newParams });
                  }}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={1}>
                <IconButton
                  color="error"
                  onClick={() => {
                    const newParams = formData.parameters.filter(
                      (_, i) => i !== index
                    );
                    setFormData({ ...formData, parameters: newParams });
                  }}
                >
                  <DeleteIcon />
                </IconButton>
              </Grid>
            </Grid>
          ))}
          <Button
            startIcon={<AddIcon />}
            onClick={() =>
              setFormData({
                ...formData,
                parameters: [
                  ...formData.parameters,
                  {
                    name: '',
                    type: 'string',
                    required: true,
                    description: '',
                  },
                ],
              })
            }
          >
            Add Parameter
          </Button>

          {/* Responses Section */}
          <Typography variant="subtitle1" gutterBottom>
            Responses
          </Typography>
          {formData.responses.map((response, index) => (
            <Grid container spacing={2} key={index}>
              <Grid item xs={12} md={2}>
                <TextField
                  label="Code"
                  type="number"
                  value={response.code}
                  onChange={(e) => {
                    const newResponses = [...formData.responses];
                    newResponses[index].code = parseInt(e.target.value);
                    setFormData({ ...formData, responses: newResponses });
                  }}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={9}>
                <TextField
                  label="Description"
                  value={response.description}
                  onChange={(e) => {
                    const newResponses = [...formData.responses];
                    newResponses[index].description = e.target.value;
                    setFormData({ ...formData, responses: newResponses });
                  }}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={1}>
                <IconButton
                  color="error"
                  onClick={() => {
                    const newResponses = formData.responses.filter(
                      (_, i) => i !== index
                    );
                    setFormData({ ...formData, responses: newResponses });
                  }}
                >
                  <DeleteIcon />
                </IconButton>
              </Grid>
            </Grid>
          ))}
          <Button
            startIcon={<AddIcon />}
            onClick={() =>
              setFormData({
                ...formData,
                responses: [
                  ...formData.responses,
                  {
                    code: 200,
                    description: 'Success',
                    schema: {},
                  },
                ],
              })
            }
          >
            Add Response
          </Button>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!formData.name || !formData.path}
        >
          {endpoint ? 'Update' : 'Create'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

function APIKeyDialog({ open, onClose, onSave }: APIKeyDialogProps) {
  const [name, setName] = useState('');
  const [expiresAt, setExpiresAt] = useState('');

  const handleSave = () => {
    onSave({ name, expires_at: expiresAt || undefined });
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>Create API Key</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <TextField
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            fullWidth
            required
          />
          <TextField
            label="Expires At"
            type="datetime-local"
            value={expiresAt}
            onChange={(e) => setExpiresAt(e.target.value)}
            fullWidth
            InputLabelProps={{
              shrink: true,
            }}
          />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained" disabled={!name}>
          Create
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelAPI() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [selectedTab, setSelectedTab] = useState(0);
  const [endpointDialogOpen, setEndpointDialogOpen] = useState(false);
  const [selectedEndpoint, setSelectedEndpoint] = useState<APIEndpoint>();
  const [apiKeyDialogOpen, setApiKeyDialogOpen] = useState(false);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const [testParameters, setTestParameters] = useState<Record<string, any>>({});

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch endpoints
  const { data: endpoints } = useQuery({
    queryKey: ['model', modelId, 'endpoints'],
    queryFn: () => modelApi.getModelEndpoints(modelId!),
    enabled: !!modelId,
  });

  // Fetch API keys
  const { data: apiKeys } = useQuery({
    queryKey: ['model', modelId, 'api-keys'],
    queryFn: () => modelApi.getModelApiKeys(modelId!),
    enabled: !!modelId,
  });

  // Create endpoint mutation
  const { mutate: createEndpoint } = useMutation({
    mutationFn: (data: Partial<APIEndpoint>) =>
      modelApi.createModelEndpoint(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'endpoints']);
      setMessage('Endpoint created successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to create endpoint');
      setSeverity('error');
    },
  });

  // Update endpoint mutation
  const { mutate: updateEndpoint } = useMutation({
    mutationFn: (data: Partial<APIEndpoint>) =>
      modelApi.updateModelEndpoint(modelId!, data.id!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'endpoints']);
      setMessage('Endpoint updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update endpoint');
      setSeverity('error');
    },
  });

  // Delete endpoint mutation
  const { mutate: deleteEndpoint } = useMutation({
    mutationFn: (endpointId: string) =>
      modelApi.deleteModelEndpoint(modelId!, endpointId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'endpoints']);
      setMessage('Endpoint deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete endpoint');
      setSeverity('error');
    },
  });

  // Create API key mutation
  const { mutate: createApiKey } = useMutation({
    mutationFn: (data: { name: string; expires_at?: string }) =>
      modelApi.createModelApiKey(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'api-keys']);
      setMessage('API key created successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to create API key');
      setSeverity('error');
    },
  });

  // Revoke API key mutation
  const { mutate: revokeApiKey } = useMutation({
    mutationFn: (keyId: string) => modelApi.revokeModelApiKey(modelId!, keyId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'api-keys']);
      setMessage('API key revoked successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to revoke API key');
      setSeverity('error');
    },
  });

  // Test endpoint mutation
  const { mutate: testEndpoint, isLoading: isTesting } = useMutation({
    mutationFn: (data: { endpointId: string; parameters: any }) =>
      modelApi.testModelEndpoint(modelId!, data.endpointId, data.parameters),
    onSuccess: () => {
      setMessage('Test completed successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Test failed');
      setSeverity('error');
    },
  });

  if (!model || !endpoints || !apiKeys) {
    return null;
  }

  const handleEndpointSave = (data: Partial<APIEndpoint>) => {
    if (selectedEndpoint) {
      updateEndpoint({ ...data, id: selectedEndpoint.id });
    } else {
      createEndpoint(data);
    }
    setEndpointDialogOpen(false);
    setSelectedEndpoint(undefined);
  };

  const handleEndpointDelete = (endpointId: string) => {
    if (
      confirm(
        'Are you sure you want to delete this endpoint? This action cannot be undone.'
      )
    ) {
      deleteEndpoint(endpointId);
    }
  };

  const handleApiKeySave = (data: { name: string; expires_at?: string }) => {
    createApiKey(data);
    setApiKeyDialogOpen(false);
  };

  const handleApiKeyRevoke = (keyId: string) => {
    if (
      confirm(
        'Are you sure you want to revoke this API key? This action cannot be undone.'
      )
    ) {
      revokeApiKey(keyId);
    }
  };

  const handleTest = (endpoint: APIEndpoint) => {
    testEndpoint({
      endpointId: endpoint.id,
      parameters: testParameters[endpoint.id] || {},
    });
  };

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET':
        return 'success';
      case 'POST':
        return 'info';
      case 'PUT':
        return 'warning';
      case 'DELETE':
        return 'error';
      default:
        return 'default';
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setMessage('Copied to clipboard');
    setSeverity('success');
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          API Management
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage API endpoints and keys for {model.name}
        </Typography>
      </Box>

      {/* Content */}
      <Card>
        <CardContent>
          <Tabs
            value={selectedTab}
            onChange={(_, newValue) => setSelectedTab(newValue)}
            aria-label="api management tabs"
          >
            <Tab
              icon={<CodeIcon />}
              label="Endpoints"
              id="api-tab-0"
              aria-controls="api-tabpanel-0"
            />
            <Tab
              icon={<KeyIcon />}
              label="API Keys"
              id="api-tab-1"
              aria-controls="api-tabpanel-1"
            />
          </Tabs>

          {/* Endpoints Tab */}
          {selectedTab === 0 && (
            <Box sx={{ mt: 3 }}>
              <Stack
                direction="row"
                spacing={2}
                alignItems="center"
                justifyContent="space-between"
                sx={{ mb: 3 }}
              >
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => {
                    setSelectedEndpoint(undefined);
                    setEndpointDialogOpen(true);
                  }}
                >
                  Create Endpoint
                </Button>
                <Tooltip title="Refresh">
                  <IconButton
                    onClick={() =>
                      queryClient.invalidateQueries([
                        'model',
                        modelId,
                        'endpoints',
                      ])
                    }
                  >
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              </Stack>

              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Method</TableCell>
                      <TableCell>Path</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>Authentication</TableCell>
                      <TableCell>Rate Limit</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {endpoints.map((endpoint) => (
                      <TableRow key={endpoint.id}>
                        <TableCell>
                          <Chip
                            label={endpoint.method}
                            color={getMethodColor(endpoint.method)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{endpoint.path}</TableCell>
                        <TableCell>{endpoint.description}</TableCell>
                        <TableCell>
                          {endpoint.authentication ? (
                            <Chip
                              icon={<SecurityIcon />}
                              label="Required"
                              color="warning"
                              size="small"
                            />
                          ) : (
                            <Chip label="None" size="small" />
                          )}
                        </TableCell>
                        <TableCell>
                          <Chip
                            icon={<PerformanceIcon />}
                            label={`${endpoint.rate_limit}/min`}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right">
                          <Stack
                            direction="row"
                            spacing={1}
                            justifyContent="flex-end"
                          >
                            <Tooltip title="Test">
                              <IconButton
                                color="primary"
                                onClick={() => handleTest(endpoint)}
                                disabled={isTesting}
                              >
                                <TestIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Edit">
                              <IconButton
                                onClick={() => {
                                  setSelectedEndpoint(endpoint);
                                  setEndpointDialogOpen(true);
                                }}
                              >
                                <EditIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Delete">
                              <IconButton
                                color="error"
                                onClick={() =>
                                  handleEndpointDelete(endpoint.id)
                                }
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
            </Box>
          )}

          {/* API Keys Tab */}
          {selectedTab === 1 && (
            <Box sx={{ mt: 3 }}>
              <Stack
                direction="row"
                spacing={2}
                alignItems="center"
                justifyContent="space-between"
                sx={{ mb: 3 }}
              >
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => setApiKeyDialogOpen(true)}
                >
                  Create API Key
                </Button>
                <Tooltip title="Refresh">
                  <IconButton
                    onClick={() =>
                      queryClient.invalidateQueries([
                        'model',
                        modelId,
                        'api-keys',
                      ])
                    }
                  >
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              </Stack>

              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Key</TableCell>
                      <TableCell>Created</TableCell>
                      <TableCell>Expires</TableCell>
                      <TableCell>Last Used</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {apiKeys.map((key) => (
                      <TableRow key={key.id}>
                        <TableCell>{key.name}</TableCell>
                        <TableCell>
                          <Stack direction="row" spacing={1} alignItems="center">
                            <Typography
                              variant="body2"
                              sx={{
                                fontFamily: 'monospace',
                                bgcolor: 'action.hover',
                                p: 0.5,
                                borderRadius: 1,
                              }}
                            >
                              {key.key.slice(0, 8)}...
                              {key.key.slice(-8)}
                            </Typography>
                            <Tooltip title="Copy">
                              <IconButton
                                size="small"
                                onClick={() => copyToClipboard(key.key)}
                              >
                                <CopyIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Stack>
                        </TableCell>
                        <TableCell>
                          {format(new Date(key.created_at), 'PPp')}
                        </TableCell>
                        <TableCell>
                          {key.expires_at
                            ? format(new Date(key.expires_at), 'PPp')
                            : 'Never'}
                        </TableCell>
                        <TableCell>
                          {key.last_used_at
                            ? format(new Date(key.last_used_at), 'PPp')
                            : 'Never'}
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={key.status}
                            color={
                              key.status === 'active'
                                ? 'success'
                                : key.status === 'expired'
                                ? 'warning'
                                : 'error'
                            }
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right">
                          <Tooltip title="Revoke">
                            <IconButton
                              color="error"
                              onClick={() => handleApiKeyRevoke(key.id)}
                              disabled={key.status !== 'active'}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Endpoint Dialog */}
      <EndpointDialog
        open={endpointDialogOpen}
        onClose={() => {
          setEndpointDialogOpen(false);
          setSelectedEndpoint(undefined);
        }}
        onSave={handleEndpointSave}
        endpoint={selectedEndpoint}
      />

      {/* API Key Dialog */}
      <APIKeyDialog
        open={apiKeyDialogOpen}
        onClose={() => setApiKeyDialogOpen(false)}
        onSave={handleApiKeySave}
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