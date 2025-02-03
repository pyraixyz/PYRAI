import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Paper,
  Stack,
  Button,
  TextField,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Snackbar,
  Grid,
  Tooltip,
  Autocomplete,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Label as LabelIcon,
  Category as CategoryIcon,
  Save as SaveIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi } from '@/api/client';

interface Tag {
  id: string;
  name: string;
  color: string;
  category: string;
  description?: string;
  created_at: string;
}

interface Category {
  id: string;
  name: string;
  description?: string;
  created_at: string;
}

interface TagDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: any) => void;
  tag?: Tag;
  categories: Category[];
}

function TagDialog({ open, onClose, onSave, tag, categories }: TagDialogProps) {
  const [formData, setFormData] = useState(
    tag || {
      name: '',
      color: '#1976d2',
      category: '',
      description: '',
    }
  );

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{tag ? 'Edit Tag' : 'Create Tag'}</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <TextField
            label="Tag Name"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            fullWidth
            required
          />

          <FormControl fullWidth>
            <InputLabel>Category</InputLabel>
            <Select
              value={formData.category}
              label="Category"
              onChange={(e) =>
                setFormData({ ...formData, category: e.target.value })
              }
            >
              {categories.map((category) => (
                <MenuItem key={category.id} value={category.id}>
                  {category.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            label="Color"
            type="color"
            value={formData.color}
            onChange={(e) => setFormData({ ...formData, color: e.target.value })}
            fullWidth
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
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!formData.name || !formData.category}
        >
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
}

interface CategoryDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: any) => void;
  category?: Category;
}

function CategoryDialog({
  open,
  onClose,
  onSave,
  category,
}: CategoryDialogProps) {
  const [formData, setFormData] = useState(
    category || {
      name: '',
      description: '',
    }
  );

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{category ? 'Edit Category' : 'Create Category'}</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <TextField
            label="Category Name"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
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
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained" disabled={!formData.name}>
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelTags() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [tagDialogOpen, setTagDialogOpen] = useState(false);
  const [categoryDialogOpen, setCategoryDialogOpen] = useState(false);
  const [selectedTag, setSelectedTag] = useState<Tag | undefined>();
  const [selectedCategory, setSelectedCategory] = useState<Category | undefined>();
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch tags
  const { data: tags } = useQuery({
    queryKey: ['model', modelId, 'tags'],
    queryFn: () => modelApi.getTags(modelId!),
    enabled: !!modelId,
  });

  // Fetch categories
  const { data: categories } = useQuery({
    queryKey: ['categories'],
    queryFn: () => modelApi.getCategories(),
  });

  // Create tag mutation
  const { mutate: createTag } = useMutation({
    mutationFn: (data: any) => modelApi.createTag(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'tags']);
      setMessage('Tag created successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to create tag');
      setSeverity('error');
    },
  });

  // Update tag mutation
  const { mutate: updateTag } = useMutation({
    mutationFn: (data: any) => modelApi.updateTag(modelId!, data.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'tags']);
      setMessage('Tag updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update tag');
      setSeverity('error');
    },
  });

  // Delete tag mutation
  const { mutate: deleteTag } = useMutation({
    mutationFn: (tagId: string) => modelApi.deleteTag(modelId!, tagId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'tags']);
      setMessage('Tag deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete tag');
      setSeverity('error');
    },
  });

  // Create category mutation
  const { mutate: createCategory } = useMutation({
    mutationFn: (data: any) => modelApi.createCategory(data),
    onSuccess: () => {
      queryClient.invalidateQueries(['categories']);
      setMessage('Category created successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to create category');
      setSeverity('error');
    },
  });

  // Update category mutation
  const { mutate: updateCategory } = useMutation({
    mutationFn: (data: any) => modelApi.updateCategory(data.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['categories']);
      setMessage('Category updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update category');
      setSeverity('error');
    },
  });

  // Delete category mutation
  const { mutate: deleteCategory } = useMutation({
    mutationFn: (categoryId: string) => modelApi.deleteCategory(categoryId),
    onSuccess: () => {
      queryClient.invalidateQueries(['categories']);
      setMessage('Category deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete category');
      setSeverity('error');
    },
  });

  if (!model || !tags || !categories) {
    return null;
  }

  const handleSaveTag = (data: any) => {
    if (selectedTag) {
      updateTag({ ...data, id: selectedTag.id });
    } else {
      createTag(data);
    }
  };

  const handleDeleteTag = (tagId: string) => {
    if (confirm('Are you sure you want to delete this tag?')) {
      deleteTag(tagId);
    }
  };

  const handleSaveCategory = (data: any) => {
    if (selectedCategory) {
      updateCategory({ ...data, id: selectedCategory.id });
    } else {
      createCategory(data);
    }
  };

  const handleDeleteCategory = (categoryId: string) => {
    if (confirm('Are you sure you want to delete this category?')) {
      deleteCategory(categoryId);
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Tags
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage tags and categories for {model.name}
        </Typography>
      </Box>

      {/* Categories Section */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={3}>
            <Stack
              direction="row"
              spacing={2}
              alignItems="center"
              justifyContent="space-between"
            >
              <Typography variant="h6">Categories</Typography>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => {
                  setSelectedCategory(undefined);
                  setCategoryDialogOpen(true);
                }}
              >
                Add Category
              </Button>
            </Stack>

            <Grid container spacing={2}>
              {categories.map((category) => (
                <Grid item xs={12} sm={6} md={4} key={category.id}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Stack spacing={1}>
                      <Stack
                        direction="row"
                        spacing={1}
                        alignItems="center"
                        justifyContent="space-between"
                      >
                        <Stack direction="row" spacing={1} alignItems="center">
                          <CategoryIcon color="primary" />
                          <Typography variant="subtitle1">
                            {category.name}
                          </Typography>
                        </Stack>
                        <Stack direction="row" spacing={1}>
                          <Tooltip title="Edit">
                            <IconButton
                              size="small"
                              onClick={() => {
                                setSelectedCategory(category);
                                setCategoryDialogOpen(true);
                              }}
                            >
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleDeleteCategory(category.id)}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      </Stack>
                      {category.description && (
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          sx={{
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                          }}
                        >
                          {category.description}
                        </Typography>
                      )}
                    </Stack>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Stack>
        </CardContent>
      </Card>

      {/* Tags Section */}
      <Card>
        <CardContent>
          <Stack spacing={3}>
            <Stack
              direction="row"
              spacing={2}
              alignItems="center"
              justifyContent="space-between"
            >
              <Typography variant="h6">Tags</Typography>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => {
                  setSelectedTag(undefined);
                  setTagDialogOpen(true);
                }}
              >
                Add Tag
              </Button>
            </Stack>

            <Grid container spacing={2}>
              {tags.map((tag) => (
                <Grid item xs={12} sm={6} md={4} key={tag.id}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Stack spacing={1}>
                      <Stack
                        direction="row"
                        spacing={1}
                        alignItems="center"
                        justifyContent="space-between"
                      >
                        <Stack direction="row" spacing={1} alignItems="center">
                          <LabelIcon sx={{ color: tag.color }} />
                          <Typography variant="subtitle1">{tag.name}</Typography>
                        </Stack>
                        <Stack direction="row" spacing={1}>
                          <Tooltip title="Edit">
                            <IconButton
                              size="small"
                              onClick={() => {
                                setSelectedTag(tag);
                                setTagDialogOpen(true);
                              }}
                            >
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleDeleteTag(tag.id)}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      </Stack>
                      <Chip
                        label={
                          categories.find((c) => c.id === tag.category)?.name ||
                          'Uncategorized'
                        }
                        size="small"
                      />
                      {tag.description && (
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          sx={{
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                          }}
                        >
                          {tag.description}
                        </Typography>
                      )}
                    </Stack>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Stack>
        </CardContent>
      </Card>

      {/* Tag Dialog */}
      <TagDialog
        open={tagDialogOpen}
        onClose={() => {
          setTagDialogOpen(false);
          setSelectedTag(undefined);
        }}
        onSave={handleSaveTag}
        tag={selectedTag}
        categories={categories}
      />

      {/* Category Dialog */}
      <CategoryDialog
        open={categoryDialogOpen}
        onClose={() => {
          setCategoryDialogOpen(false);
          setSelectedCategory(undefined);
        }}
        onSave={handleSaveCategory}
        category={selectedCategory}
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