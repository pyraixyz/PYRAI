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
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Snackbar,
  IconButton,
  Tooltip,
  Card,
  CardContent,
  Autocomplete,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon,
  Label as TagIcon,
  Category as CategoryIcon,
  LocalOffer as OfferIcon,
  Bookmark as BookmarkIcon,
  ColorLens as ColorIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { modelApi } from '@/api/client';

interface Tag {
  id: string;
  name: string;
  category: string;
  color?: string;
  description?: string;
  created_at: string;
  updated_at: string;
  created_by: {
    id: string;
    name: string;
  };
}

interface Category {
  id: string;
  name: string;
  description?: string;
  color?: string;
  tag_count: number;
  created_at: string;
  updated_at: string;
}

interface TagDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: Partial<Tag>) => void;
  tag?: Tag;
  categories: Category[];
}

interface CategoryDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: Partial<Category>) => void;
  category?: Category;
}

const COLORS = [
  '#f44336',
  '#e91e63',
  '#9c27b0',
  '#673ab7',
  '#3f51b5',
  '#2196f3',
  '#03a9f4',
  '#00bcd4',
  '#009688',
  '#4caf50',
  '#8bc34a',
  '#cddc39',
  '#ffeb3b',
  '#ffc107',
  '#ff9800',
  '#ff5722',
];

function TagDialog({
  open,
  onClose,
  onSave,
  tag,
  categories,
}: TagDialogProps) {
  const [formData, setFormData] = useState(
    tag || {
      name: '',
      category: '',
      color: COLORS[0],
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
            label="Name"
            value={formData.name}
            onChange={(e) =>
              setFormData({ ...formData, name: e.target.value })
            }
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
            <InputLabel>Color</InputLabel>
            <Select
              value={formData.color}
              label="Color"
              onChange={(e) =>
                setFormData({ ...formData, color: e.target.value })
              }
              renderValue={(color) => (
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box
                    sx={{
                      width: 20,
                      height: 20,
                      borderRadius: 1,
                      bgcolor: color,
                      mr: 1,
                    }}
                  />
                  {color}
                </Box>
              )}
            >
              {COLORS.map((color) => (
                <MenuItem key={color} value={color}>
                  <Box
                    sx={{
                      width: 20,
                      height: 20,
                      borderRadius: 1,
                      bgcolor: color,
                      mr: 1,
                    }}
                  />
                  {color}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!formData.name || !formData.category}
        >
          {tag ? 'Update' : 'Create'}
        </Button>
      </DialogActions>
    </Dialog>
  );
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
      color: COLORS[0],
    }
  );

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {category ? 'Edit Category' : 'Create Category'}
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
            <InputLabel>Color</InputLabel>
            <Select
              value={formData.color}
              label="Color"
              onChange={(e) =>
                setFormData({ ...formData, color: e.target.value })
              }
              renderValue={(color) => (
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box
                    sx={{
                      width: 20,
                      height: 20,
                      borderRadius: 1,
                      bgcolor: color,
                      mr: 1,
                    }}
                  />
                  {color}
                </Box>
              )}
            >
              {COLORS.map((color) => (
                <MenuItem key={color} value={color}>
                  <Box
                    sx={{
                      width: 20,
                      height: 20,
                      borderRadius: 1,
                      bgcolor: color,
                      mr: 1,
                    }}
                  />
                  {color}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!formData.name}
        >
          {category ? 'Update' : 'Create'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelTags() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [tagDialogOpen, setTagDialogOpen] = useState(false);
  const [selectedTag, setSelectedTag] = useState<Tag>();
  const [categoryDialogOpen, setCategoryDialogOpen] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<Category>();
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
    queryFn: () => modelApi.getModelTags(modelId!),
    enabled: !!modelId,
  });

  // Fetch categories
  const { data: categories } = useQuery({
    queryKey: ['model', modelId, 'categories'],
    queryFn: () => modelApi.getModelCategories(modelId!),
    enabled: !!modelId,
  });

  // Create tag mutation
  const { mutate: createTag } = useMutation({
    mutationFn: (data: Partial<Tag>) =>
      modelApi.createModelTag(modelId!, data),
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
    mutationFn: (data: Partial<Tag>) =>
      modelApi.updateModelTag(modelId!, data.id!, data),
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
    mutationFn: (tagId: string) =>
      modelApi.deleteModelTag(modelId!, tagId),
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
    mutationFn: (data: Partial<Category>) =>
      modelApi.createModelCategory(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'categories']);
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
    mutationFn: (data: Partial<Category>) =>
      modelApi.updateModelCategory(modelId!, data.id!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'categories']);
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
    mutationFn: (categoryId: string) =>
      modelApi.deleteModelCategory(modelId!, categoryId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'categories']);
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

  const handleTagSave = (data: Partial<Tag>) => {
    if (selectedTag) {
      updateTag({ ...data, id: selectedTag.id });
    } else {
      createTag(data);
    }
    setTagDialogOpen(false);
    setSelectedTag(undefined);
  };

  const handleTagDelete = (tagId: string) => {
    if (
      confirm(
        'Are you sure you want to delete this tag? This action cannot be undone.'
      )
    ) {
      deleteTag(tagId);
    }
  };

  const handleCategorySave = (data: Partial<Category>) => {
    if (selectedCategory) {
      updateCategory({ ...data, id: selectedCategory.id });
    } else {
      createCategory(data);
    }
    setCategoryDialogOpen(false);
    setSelectedCategory(undefined);
  };

  const handleCategoryDelete = (categoryId: string) => {
    if (
      confirm(
        'Are you sure you want to delete this category? This will also delete all tags in this category.'
      )
    ) {
      deleteCategory(categoryId);
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Tags & Categories
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage tags and categories for {model.name}
        </Typography>
      </Box>

      {/* Content */}
      <Grid container spacing={3}>
        {/* Categories */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Stack
                direction="row"
                spacing={2}
                alignItems="center"
                justifyContent="space-between"
                sx={{ mb: 2 }}
              >
                <Typography variant="h6">Categories</Typography>
                <Button
                  variant="contained"
                  size="small"
                  startIcon={<AddIcon />}
                  onClick={() => {
                    setSelectedCategory(undefined);
                    setCategoryDialogOpen(true);
                  }}
                >
                  Add
                </Button>
              </Stack>

              <List>
                {categories.map((category) => (
                  <React.Fragment key={category.id}>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Stack direction="row" spacing={1} alignItems="center">
                            <Box
                              sx={{
                                width: 16,
                                height: 16,
                                borderRadius: 1,
                                bgcolor: category.color,
                              }}
                            />
                            <Typography>{category.name}</Typography>
                          </Stack>
                        }
                        secondary={`${category.tag_count} tags`}
                      />
                      <ListItemSecondaryAction>
                        <Stack direction="row" spacing={1}>
                          <Tooltip title="Edit">
                            <IconButton
                              edge="end"
                              onClick={() => {
                                setSelectedCategory(category);
                                setCategoryDialogOpen(true);
                              }}
                            >
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              edge="end"
                              color="error"
                              onClick={() => handleCategoryDelete(category.id)}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      </ListItemSecondaryAction>
                    </ListItem>
                    <Divider />
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Tags */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Stack
                direction="row"
                spacing={2}
                alignItems="center"
                justifyContent="space-between"
                sx={{ mb: 2 }}
              >
                <Typography variant="h6">Tags</Typography>
                <Stack direction="row" spacing={1}>
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={<AddIcon />}
                    onClick={() => {
                      setSelectedTag(undefined);
                      setTagDialogOpen(true);
                    }}
                  >
                    Add
                  </Button>
                  <Tooltip title="Refresh">
                    <IconButton
                      size="small"
                      onClick={() =>
                        queryClient.invalidateQueries([
                          'model',
                          modelId,
                          'tags',
                        ])
                      }
                    >
                      <RefreshIcon />
                    </IconButton>
                  </Tooltip>
                </Stack>
              </Stack>

              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {tags.map((tag) => {
                  const category = categories.find(
                    (c) => c.id === tag.category
                  );
                  return (
                    <Chip
                      key={tag.id}
                      label={tag.name}
                      sx={{
                        bgcolor: tag.color,
                        '& .MuiChip-deleteIcon': {
                          color: 'inherit',
                        },
                      }}
                      onDelete={() => handleTagDelete(tag.id)}
                      onClick={() => {
                        setSelectedTag(tag);
                        setTagDialogOpen(true);
                      }}
                    />
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tag Dialog */}
      <TagDialog
        open={tagDialogOpen}
        onClose={() => {
          setTagDialogOpen(false);
          setSelectedTag(undefined);
        }}
        onSave={handleTagSave}
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
        onSave={handleCategorySave}
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