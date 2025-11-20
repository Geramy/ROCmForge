//! Tests for TensorShape stride computation

use rocmforge::loader::mmap_loader::TensorShape;

#[test]
fn test_tensor_shape_from_dims_1d() {
    let dims = vec![100];
    let shape = TensorShape::from_dims(&dims);

    assert_eq!(shape.dims(), &dims);
    assert_eq!(shape.strides(), &[1]); // stride[0] = 1
}

#[test]
fn test_tensor_shape_from_dims_2d() {
    let dims = vec![64, 128];
    let shape = TensorShape::from_dims(&dims);

    assert_eq!(shape.dims(), &dims);
    assert_eq!(shape.strides(), &[128, 1]); // stride[0] = 128, stride[1] = 1
}

#[test]
fn test_tensor_shape_from_dims_3d() {
    let dims = vec![10, 20, 30];
    let shape = TensorShape::from_dims(&dims);

    assert_eq!(shape.dims(), &dims);
    assert_eq!(shape.strides(), &[600, 30, 1]); // stride[0] = 20*30, stride[1] = 30, stride[2] = 1
}

#[test]
fn test_tensor_shape_from_dims_4d() {
    let dims = vec![2, 3, 4, 5];
    let shape = TensorShape::from_dims(&dims);

    assert_eq!(shape.dims(), &dims);
    assert_eq!(shape.strides(), &[60, 20, 5, 1]); // stride[0] = 3*4*5, stride[1] = 4*5, stride[2] = 5, stride[3] = 1
}

#[test]
fn test_tensor_shape_empty_dims() {
    let dims = vec![];
    let shape = TensorShape::from_dims(&dims);

    assert_eq!(shape.dims(), &dims);
    assert_eq!(shape.strides(), &[] as &[usize]); // No strides for empty shape
}

#[test]
fn test_tensor_shape_single_element() {
    let dims = vec![1];
    let shape = TensorShape::from_dims(&dims);

    assert_eq!(shape.dims(), &dims);
    assert_eq!(shape.strides(), &[1]);
}

#[test]
fn test_tensor_shape_compute_strides_consistency() {
    // Test that stride computation is consistent with row-major layout
    let test_cases = vec![
        (vec![5], vec![1]),
        (vec![3, 4], vec![4, 1]),
        (vec![2, 3, 4], vec![12, 4, 1]),
        (vec![1, 2, 3, 4], vec![24, 12, 4, 1]),
    ];

    for (dims, expected_strides) in test_cases {
        let shape = TensorShape::from_dims(&dims);
        assert_eq!(shape.dims(), &dims);
        assert_eq!(shape.strides(), &expected_strides);
    }
}

#[test]
fn test_tensor_shape_total_elements() {
    let test_cases = vec![
        (vec![5], 5),
        (vec![3, 4], 12),
        (vec![2, 3, 4], 24),
        (vec![1, 2, 3, 4], 24),
        (vec![], 1), // Empty shape treated as 1 element
    ];

    for (dims, expected_total) in test_cases {
        let shape = TensorShape::from_dims(&dims);
        let total: usize = shape.dims().iter().product();
        assert_eq!(total, expected_total);
    }
}
