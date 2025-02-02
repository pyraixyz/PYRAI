import { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Stack,
  Chip,
} from '@mui/material';
import {
  Add as AddIcon,
  Upload as UploadIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { formatBytes } from '@/utils/format';
import { modelApi } from '@/api/client';

interface ModelDialogProps {
  open: boolean;
  onClose: () => void;
  model?: Model;
}

function ModelDialog({ open, onClose, model }: ModelDialogProps) {
  const [name, setName] = useState(model?.name || '');
  const [description, setDescription] = useState(model?.description || '');
  const queryClient = useQueryClient();

  const { mutate: saveModel, isLoading } = useMutation({
    mutationFn: model
      ? (data: Partial<Model>) => modelApi.updateModel(model.id, data)
      : modelApi.createModel,
    onSuccess: () => {
      queryClient.invalidateQueries(['models']);
      onClose();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    saveModel({ name, description });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>
          {model ? 'Edit Model' : 'Create New Model'}
        </DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 2 }}>
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
              fullWidth
              multiline
              rows={3}
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

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
  modelId: string;
}

function UploadDialog({ open, onClose, modelId }: UploadDialogProps) {
  const [file, setFile] = useState<File | null>(null);
  const queryClient = useQueryClient();

  const { mutate: uploadVersion, isLoading } = useMutation({
    mutationFn: (data: FormData) => modelApi.uploadVersion(modelId, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['models']);
      onClose();
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    uploadVersion(formData);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>Upload Model Version</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 2 }}>
            <Button
              variant="outlined"
              component="label"
              startIcon={<UploadIcon />}
            >
              Choose File
              <input
                type="file"
                hidden
                accept=".pth,.onnx,.h5,.pb"
                onChange={handleFileChange}
              />
            </Button>
            {file && (
              <Typography variant="body2" color="text.secondary">
                Selected file: {file.name} ({formatBytes(file.size)})
              </Typography>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            disabled={isLoading || !file}
          >
            {isLoading ? 'Uploading...' : 'Upload'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
}

export default function Models() {
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const queryClient = useQueryClient();

  // Fetch models
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: modelApi.listModels,
  });

  // Delete model mutation
  const { mutate: deleteModel } = useMutation({
    mutationFn: modelApi.deleteModel,
    onSuccess: () => {
      queryClient.invalidateQueries(['models']);
    },
  });

  const handleEdit = (model: Model) => {
    setSelectedModel(model);
    setModelDialogOpen(true);
  };

  const handleUpload = (model: Model) => {
    setSelectedModel(model);
    setUploadDialogOpen(true);
  };

  const handleDelete = (modelId: string) => {
    if (confirm('Are you sure you want to delete this model?')) {
      deleteModel(modelId);
    }
  };

  const handleDownload = (model: Model, version: ModelVersion) => {
    window.open(version.download_url, '_blank');
  };

  if (!models) {
    return null;
  }

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
        <Typography variant="h5">Models</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            setSelectedModel(null);
            setModelDialogOpen(true);
          }}
        >
          New Model
        </Button>
      </Box>

      {/* Models List */}
      <Grid container spacing={3}>
        {models.map((model) => (
          <Grid item xs={12} key={model.id}>
            <Card>
              <CardContent>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    mb: 2,
                  }}
                >
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      {model.name}
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 1 }}
                    >
                      {model.description}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Created at:{' '}
                      {format(new Date(model.created_at), 'yyyy-MM-dd HH:mm:ss')}
                    </Typography>
                  </Box>
                  <Stack direction="row" spacing={1}>
                    <Tooltip title="Upload Version">
                      <IconButton
                        size="small"
                        onClick={() => handleUpload(model)}
                      >
                        <UploadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit">
                      <IconButton
                        size="small"
                        onClick={() => handleEdit(model)}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDelete(model.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </Box>

                {/* Versions Table */}
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Version</TableCell>
                        <TableCell>Size</TableCell>
                        <TableCell>Upload Time</TableCell>
                        <TableCell>Format</TableCell>
                        <TableCell align="right">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {model.versions.map((version) => (
                        <TableRow key={version.id}>
                          <TableCell>
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Typography variant="body2">
                                v{version.version}
                              </Typography>
                              {version.is_latest && (
                                <Chip
                                  label="Latest"
                                  size="small"
                                  color="primary"
                                />
                              )}
                            </Stack>
                          </TableCell>
                          <TableCell>
                            {formatBytes(version.file_size)}
                          </TableCell>
                          <TableCell>
                            {format(
                              new Date(version.uploaded_at),
                              'yyyy-MM-dd HH:mm:ss'
                            )}
                          </TableCell>
                          <TableCell>{version.format}</TableCell>
                          <TableCell align="right">
                            <Tooltip title="Download">
                              <IconButton
                                size="small"
                                onClick={() => handleDownload(model, version)}
                              >
                                <DownloadIcon />
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Dialogs */}
      <ModelDialog
        open={modelDialogOpen}
        onClose={() => {
          setModelDialogOpen(false);
          setSelectedModel(null);
        }}
        model={selectedModel || undefined}
      />
      {selectedModel && (
        <UploadDialog
          open={uploadDialogOpen}
          onClose={() => {
            setUploadDialogOpen(false);
            setSelectedModel(null);
          }}
          modelId={selectedModel.id}
        />
      )}
    </Box>
  );
} 