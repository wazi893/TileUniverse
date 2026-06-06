use engine::block_sparse_state::BlockSparseState;

#[test]
fn test_block_sparse_compiles() {
    let state = BlockSparseState::new_zero(10);
    assert_eq!(state.block_count(), 1);
    assert_eq!(state.nnz(), 1);
}
