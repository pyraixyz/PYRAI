import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Stack,
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
  TextField,
  Alert,
  Snackbar,
  Paper,
  Checkbox,
  CircularProgress,
  Toolbar,
  alpha,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Compare as CompareIcon,
  Download as DownloadIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { modelApi } from '@/api/client';

interface VersionDialogProps {
  open: boolean;
  onClose: () => void;
  onUpload: (data: FormData) => void;
  isUploading: boolean;
}

function VersionDialog({ open, onClose, onUpload, isUploading }: VersionDialogProps) {
  const [file, setFile] = useState<File | null>(null);
  const [description, setDescription] = useState('');

  const handleUpload = () => {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('description', description);
    
    onUpload(formData);
    setFile(null);
    setDescription('');
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Upload New Version</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 2 }}>
          <Button
            variant="outlined"
            component="label"
            startIcon={<UploadIcon />}
          >
            Select Model File
            <input
              type="file"
              hidden
              accept=".h5,.pkl,.pt,.pth,.onnx"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </Button>
          {file && (
            <Typography variant="body2" color="text.secondary">
              Selected: {file.name}
            </Typography>
          )}
          <TextField
            label="Version Description"
            multiline
            rows={4}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleUpload}
          variant="contained"
          disabled={!file || isUploading}
        >
          {isUploading ? 'Uploading...' : 'Upload'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

interface EnhancedTableToolbarProps {
  numSelected: number;
  onDeleteSelected: () => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
}

function EnhancedTableToolbar({
  numSelected,
  onDeleteSelected,
  searchQuery,
  onSearchChange,
}: EnhancedTableToolbarProps) {
  return (
    <Toolbar
      sx={{
        pl: { sm: 2 },
        pr: { xs: 1, sm: 1 },
        ...(numSelected > 0 && {
          bgcolor: (theme) =>
            alpha(theme.palette.primary.main, theme.palette.action.activatedOpacity),
        }),
      }}
    >
      {numSelected > 0 ? (
        <Typography
          sx={{ flex: '1 1 100%' }}
          color="inherit"
          variant="subtitle1"
          component="div"
        >
          {numSelected} selected
        </Typography>
      ) : (
        <Box sx={{ flex: '1 1 100%', display: 'flex', alignItems: 'center' }}>
          <SearchIcon sx={{ color: 'action.active', mr: 1 }} />
          <TextField
            placeholder="Search versions..."
            variant="standard"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            sx={{ width: 200 }}
          />
        </Box>
      )}

      {numSelected > 0 && (
        <Tooltip title="Delete selected">
          <IconButton onClick={onDeleteSelected}>
            <DeleteIcon />
          </IconButton>
        </Tooltip>
      )}
    </Toolbar>
  );
}

export default function ModelVersions() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch model
  const { data: model, isLoading, error } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Upload version mutation
  const { mutate: uploadVersion, isLoading: isUploading } = useMutation({
    mutationFn: (data: FormData) => modelApi.uploadModelVersion(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId]);
      setUploadDialogOpen(false);
      setMessage('Version uploaded successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to upload version');
      setSeverity('error');
    },
  });

  // Delete version mutation
  const { mutate: deleteVersion } = useMutation({
    mutationFn: (versionId: string) =>
      modelApi.deleteModelVersion(modelId!, versionId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId]);
      setMessage('Version deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete version');
      setSeverity('error');
    },
  });

  // Set default version mutation
  const { mutate: setDefaultVersion } = useMutation({
    mutationFn: (versionId: string) =>
      modelApi.setDefaultModelVersion(modelId!, versionId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId]);
      setMessage('Default version updated');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update default version');
      setSeverity('error');
    },
  });

  // Add bulk delete mutation
  const { mutate: bulkDeleteVersions } = useMutation({
    mutationFn: (versionIds: string[]) =>
      Promise.all(versionIds.map((id) => modelApi.deleteModelVersion(modelId!, id))),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId]);
      setMessage(`Successfully deleted ${selectedVersions.length} versions`);
      setSeverity('success');
      setSelectedVersions([]);
    },
    onError: () => {
      setMessage('Failed to delete selected versions');
      setSeverity('error');
    },
  });

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          Failed to load model versions. Please try again later.
        </Alert>
      </Box>
    );
  }

  if (!model) return null;

  if (model.versions.length === 0) {
    return (
      <Box>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            Model Versions
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage versions for {model.name}
          </Typography>
        </Box>
        
        <Card variant="outlined">
          <CardContent>
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No Versions Available
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Upload your first model version to get started
              </Typography>
              <Button
                variant="contained"
                startIcon={<UploadIcon />}
                onClick={() => setUploadDialogOpen(true)}
              >
                Upload Version
              </Button>
            </Box>
          </CardContent>
        </Card>

        <VersionDialog
          open={uploadDialogOpen}
          onClose={() => setUploadDialogOpen(false)}
          onUpload={uploadVersion}
          isUploading={isUploading}
        />
      </Box>
    );
  }

  const handleVersionSelect = (versionId: string) => {
    setSelectedVersions((prev) => {
      if (prev.includes(versionId)) {
        return prev.filter((id) => id !== versionId);
      }
      return [...prev, versionId];
    });
  };

  const handleDelete = (versionId: string) => {
    if (
      confirm(
        'Are you sure you want to delete this version? This action cannot be undone.'
      )
    ) {
      deleteVersion(versionId);
    }
  };

  // Add handler for bulk delete
  const handleBulkDelete = () => {
    if (
      confirm(
        `Are you sure you want to delete ${selectedVersions.length} versions? This action cannot be undone.`
      )
    ) {
      bulkDeleteVersions(selectedVersions);
    }
  };

  // Add filtered versions computation
  const filteredVersions = model?.versions.filter((version) => {
    const searchLower = searchQuery.toLowerCase();
    return (
      version.version.toString().includes(searchLower) ||
      version.description.toLowerCase().includes(searchLower) ||
      version.format.toLowerCase().includes(searchLower)
    );
  }) || [];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Versions
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage versions for {model.name}
        </Typography>
      </Box>

      {/* Actions */}
      <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          onClick={() => setUploadDialogOpen(true)}
        >
          Upload Version
        </Button>
        <Button
          variant="outlined"
          startIcon={<CompareIcon />}
          disabled={selectedVersions.length < 2}
          onClick={() => {
            // Navigate to comparison page
            window.location.href = `/models/${modelId}/comparison?versions=${selectedVersions.join(',')}`;
          }}
        >
          Compare Selected
        </Button>
      </Stack>

      {/* Enhanced Table */}
      <Paper variant="outlined">
        <EnhancedTableToolbar
          numSelected={selectedVersions.length}
          onDeleteSelected={handleBulkDelete}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
        />
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={
                      selectedVersions.length > 0 &&
                      selectedVersions.length < filteredVersions.length
                    }
                    checked={
                      filteredVersions.length > 0 &&
                      selectedVersions.length === filteredVersions.length
                    }
                    onChange={(event) => {
                      if (event.target.checked) {
                        setSelectedVersions(filteredVersions.map((v) => v.id));
                      } else {
                        setSelectedVersions([]);
                      }
                    }}
                  />
                </TableCell>
                <TableCell>Version</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Size</TableCell>
                <TableCell>Upload Date</TableCell>
                <TableCell>Format</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredVersions.map((version) => (
                <TableRow key={version.id}>
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedVersions.includes(version.id)}
                      onChange={() => handleVersionSelect(version.id)}
                    />
                  </TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Typography>v{version.version}</Typography>
                      {version.is_default && (
                        <Chip label="Default" size="small" color="primary" />
                      )}
                    </Stack>
                  </TableCell>
                  <TableCell>{version.description}</TableCell>
                  <TableCell>{formatBytes(version.size)}</TableCell>
                  <TableCell>
                    {format(new Date(version.created_at), 'yyyy-MM-dd HH:mm:ss')}
                  </TableCell>
                  <TableCell>{version.format}</TableCell>
                  <TableCell align="right">
                    <Stack direction="row" spacing={1} justifyContent="flex-end">
                      <Tooltip title={version.is_default ? 'Default Version' : 'Set as Default'}>
                        <IconButton
                          size="small"
                          onClick={() => setDefaultVersion(version.id)}
                          color={version.is_default ? 'primary' : 'default'}
                        >
                          {version.is_default ? <StarIcon /> : <StarBorderIcon />}
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Download">
                        <IconButton
                          size="small"
                          onClick={() => window.open(version.download_url)}
                        >
                          <DownloadIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDelete(version.id)}
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
      </Paper>

      {/* Upload Dialog */}
      <VersionDialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        onUpload={uploadVersion}
        isUploading={isUploading}
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

// Utility function to format bytes
function formatBytes(bytes: number) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
} 