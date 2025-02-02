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
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
} from '@mui/material';
import {
  Add as AddIcon,
  Upload as UploadIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { formatBytes } from '@/utils/format';
import { datasetApi } from '@/api/client';

interface DatasetDialogProps {
  open: boolean;
  onClose: () => void;
  dataset?: Dataset;
}

function DatasetDialog({ open, onClose, dataset }: DatasetDialogProps) {
  const [name, setName] = useState(dataset?.name || '');
  const [description, setDescription] = useState(dataset?.description || '');
  const [type, setType] = useState(dataset?.type || 'image');
  const queryClient = useQueryClient();

  const { mutate: saveDataset, isLoading } = useMutation({
    mutationFn: dataset
      ? (data: Partial<Dataset>) => datasetApi.updateDataset(dataset.id, data)
      : datasetApi.createDataset,
    onSuccess: () => {
      queryClient.invalidateQueries(['datasets']);
      onClose();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    saveDataset({ name, description, type });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>
          {dataset ? 'Edit Dataset' : 'Create New Dataset'}
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
            <FormControl fullWidth>
              <InputLabel>Type</InputLabel>
              <Select
                value={type}
                label="Type"
                onChange={(e) => setType(e.target.value)}
              >
                <MenuItem value="image">Image</MenuItem>
                <MenuItem value="text">Text</MenuItem>
                <MenuItem value="audio">Audio</MenuItem>
                <MenuItem value="video">Video</MenuItem>
              </Select>
            </FormControl>
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
  datasetId: string;
}

function UploadDialog({ open, onClose, datasetId }: UploadDialogProps) {
  const [files, setFiles] = useState<FileList | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const queryClient = useQueryClient();

  const { mutate: uploadFiles } = useMutation({
    mutationFn: async (formData: FormData) => {
      setUploading(true);
      setProgress(0);
      try {
        await datasetApi.uploadFiles(datasetId, formData, (progress) => {
          setProgress(progress);
        });
      } finally {
        setUploading(false);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['datasets']);
      onClose();
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFiles(e.target.files);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!files) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    uploadFiles(formData);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>Upload Files</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 2 }}>
            <Button
              variant="outlined"
              component="label"
              startIcon={<UploadIcon />}
            >
              Choose Files
              <input
                type="file"
                hidden
                multiple
                onChange={handleFileChange}
              />
            </Button>
            {files && (
              <Typography variant="body2" color="text.secondary">
                Selected {files.length} files
              </Typography>
            )}
            {uploading && (
              <Box sx={{ width: '100%' }}>
                <LinearProgress
                  variant="determinate"
                  value={progress}
                  sx={{ mb: 1 }}
                />
                <Typography variant="body2" color="text.secondary">
                  Uploading... {Math.round(progress)}%
                </Typography>
              </Box>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            disabled={uploading || !files}
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
}

export default function Datasets() {
  const [datasetDialogOpen, setDatasetDialogOpen] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const queryClient = useQueryClient();

  // Fetch datasets
  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: datasetApi.listDatasets,
  });

  // Delete dataset mutation
  const { mutate: deleteDataset } = useMutation({
    mutationFn: datasetApi.deleteDataset,
    onSuccess: () => {
      queryClient.invalidateQueries(['datasets']);
    },
  });

  const handleEdit = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setDatasetDialogOpen(true);
  };

  const handleUpload = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setUploadDialogOpen(true);
  };

  const handleDelete = (datasetId: string) => {
    if (confirm('Are you sure you want to delete this dataset?')) {
      deleteDataset(datasetId);
    }
  };

  const handleDownload = (dataset: Dataset) => {
    window.open(dataset.download_url, '_blank');
  };

  if (!datasets) {
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
        <Typography variant="h5">Datasets</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            setSelectedDataset(null);
            setDatasetDialogOpen(true);
          }}
        >
          New Dataset
        </Button>
      </Box>

      {/* Datasets List */}
      <Grid container spacing={3}>
        {datasets.map((dataset) => (
          <Grid item xs={12} key={dataset.id}>
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
                      {dataset.name}
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 1 }}
                    >
                      {dataset.description}
                    </Typography>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Chip
                        label={dataset.type}
                        size="small"
                        color="primary"
                      />
                      <Typography variant="caption" color="text.secondary">
                        Created at:{' '}
                        {format(new Date(dataset.created_at), 'yyyy-MM-dd HH:mm:ss')}
                      </Typography>
                    </Stack>
                  </Box>
                  <Stack direction="row" spacing={1}>
                    <Tooltip title="Upload Files">
                      <IconButton
                        size="small"
                        onClick={() => handleUpload(dataset)}
                      >
                        <UploadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="View Files">
                      <IconButton size="small">
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Download">
                      <IconButton
                        size="small"
                        onClick={() => handleDownload(dataset)}
                      >
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit">
                      <IconButton
                        size="small"
                        onClick={() => handleEdit(dataset)}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDelete(dataset.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </Box>

                {/* Dataset Stats */}
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Total Files</TableCell>
                        <TableCell>Total Size</TableCell>
                        <TableCell>Processed</TableCell>
                        <TableCell>Last Updated</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>{dataset.file_count}</TableCell>
                        <TableCell>{formatBytes(dataset.total_size)}</TableCell>
                        <TableCell>
                          <Stack direction="row" spacing={1} alignItems="center">
                            <LinearProgress
                              variant="determinate"
                              value={dataset.processed_percentage * 100}
                              sx={{ width: 100 }}
                            />
                            <Typography variant="body2">
                              {Math.round(dataset.processed_percentage * 100)}%
                            </Typography>
                          </Stack>
                        </TableCell>
                        <TableCell>
                          {format(
                            new Date(dataset.updated_at),
                            'yyyy-MM-dd HH:mm:ss'
                          )}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Dialogs */}
      <DatasetDialog
        open={datasetDialogOpen}
        onClose={() => {
          setDatasetDialogOpen(false);
          setSelectedDataset(null);
        }}
        dataset={selectedDataset || undefined}
      />
      {selectedDataset && (
        <UploadDialog
          open={uploadDialogOpen}
          onClose={() => {
            setUploadDialogOpen(false);
            setSelectedDataset(null);
          }}
          datasetId={selectedDataset.id}
        />
      )}
    </Box>
  );
} 