# Code Quality Tracking

Last updated: 2026-02-10

This document tracks known code quality issues identified during code review. These are not bugs, but areas where the code could be improved for maintainability, readability, or following Rust best practices.

## ✅ Fixed Issues

### Sprint: Code Quality Cleanup (Feb 2026)
- ✅ Invalid feature flag `__never_enabled` not defined
- ✅ Excessive float precision in FRAC_1_SQRT_2 constant (5 occurrences)
- ✅ Manual modulo check instead of `.is_multiple_of()`

## 🔴 High Priority Issues

### Empty Lines After Doc Comments (3 occurrences)
**Location:** `crates/logic-fabric-core/src/algebraic_fusion.rs`
- Lines: 3113, 5008, 5205

**Issue:** Empty lines between doc comments and the items they document can cause confusion.

**Fix:** Remove empty lines or convert to regular comments.

```rust
// ❌ Bad
/// Documentation
///
pub fn foo() {}

// ✅ Good
/// Documentation
pub fn foo() {}
```

### Loop Variables Used Only for Indexing (10+ occurrences)
**Location:** `crates/logic-fabric-core/src/algebraic_fusion.rs`

**Issue:** Using loop counters only to index arrays is less idiomatic than using iterators.

**Examples:**
- Line 3388: `for i in 0..data.len()` → use `data.iter().enumerate()`
- Line 3491: Nested loops with indices
- Line 3858: `eig_idx` only used for indexing

**Fix:**
```rust
// ❌ Bad
for i in 0..data.len() {
    process(data[i]);
}

// ✅ Good
for item in data.iter() {
    process(item);
}

// ✅ Also good when index is needed
for (i, item) in data.iter().enumerate() {
    println!("Item {} is {}", i, item);
}
```

**Impact:** Medium - Makes code more idiomatic and reduces bounds checking overhead.

## 🟡 Medium Priority Issues

### Methods That Mimic Standard Traits
**Location:** Multiple files

**Issue:** Custom methods named `add`, `sub`, `mul`, `neg` can be confused with standard trait methods.

**Fix:** Implement proper traits from `std::ops`:
```rust
// ❌ Confusing
impl Foo {
    fn add(&self, other: &Self) -> Self { ... }
}

// ✅ Clear
impl std::ops::Add for Foo {
    type Output = Self;
    fn add(self, other: Self) -> Self { ... }
}
```

### Manual `RangeInclusive::contains` Implementation
**Location:** Various

**Issue:** Reimplementing standard library functionality.

**Fix:** Use `range.contains(&value)` instead of manual bounds checking.

### Unnecessary Type Casts
**Examples:** `i32` -> `i32`, `u32` -> `u32`

**Impact:** Low - Code noise, but doesn't affect functionality.

### Clamp-Like Pattern Without Using Clamp Function
**Location:** `algebraic_fusion.rs:4112`

**Fix:** Use `.clamp(min, max)` method instead of manual min/max chains.

### Usage of `contains_key` Followed by `insert`
**Issue:** Inefficient HashMap usage pattern.

**Fix:**
```rust
// ❌ Inefficient
if !map.contains_key(&key) {
    map.insert(key, value);
}

// ✅ Efficient
map.entry(key).or_insert(value);
```

## 🟢 Low Priority Issues (Code Style)

### Debug Print Statements (2,905 occurrences)
**Locations:** 99 files

**Issue:** Many `println!`, `dbg!`, `eprintln!` calls throughout the codebase.

**Analysis Needed:**
- Many are in test code (legitimate)
- Some are in examples (legitimate)
- Some might be leftover debug code in production paths

**Action:** Review production code paths for debug prints that should use proper logging.

### Dead Code Suppressions (113 occurrences)
**Locations:** 43 files with `#[allow(dead_code)]` or `#[allow(unused)]`

**Issue:** Suppressing warnings instead of addressing them.

**Action:** For each suppression:
1. Determine if code is actually needed
2. If needed, document why it appears unused
3. If not needed, remove it
4. Consider if it should be feature-gated instead

**Known Legitimate Cases:**
- `src/tile_cpu/execute.rs`: Fields for future Sprint 85-87 implementation
- `src/tile_cpu/wiring.rs`: Future infrastructure methods

### TODO/FIXME Comments (33 occurrences)
**Locations:** 19 files

**Issue:** Known technical debt markers.

**Action:** Create GitHub issues for important TODOs, remove obsolete ones.

**High-Value TODOs to Address:**
- Quantum JIT compilation improvements
- Distributed system enhancements
- Performance optimizations

## 📊 Statistics

| Category | Count | Status |
|----------|-------|--------|
| Clippy warnings (all) | ~50 | 7 fixed |
| unwrap/expect/panic | 1,355 | Many legitimate |
| Debug prints | 2,905 | Review needed |
| Dead code suppressions | 113 | Audit needed |
| TODO/FIXME comments | 33 | Tracking needed |

## 🎯 Recommended Action Plan

### Phase 1: Quick Wins (Completed ✅)
- ✅ Fix invalid feature flags
- ✅ Fix float precision issues
- ✅ Fix manual `.is_multiple_of()` implementations

### Phase 2: High Priority (Next Sprint)
- [ ] Fix empty lines after doc comments (5 min)
- [ ] Convert indexed loops to iterators (2-3 hours)
- [ ] Review and implement standard traits where appropriate (4-6 hours)

### Phase 3: Medium Priority (Future Sprint)
- [ ] Audit `#[allow(dead_code)]` suppressions
- [ ] Review debug print statements in production code
- [ ] Convert manual range checks to `.contains()` calls

### Phase 4: Continuous Improvement
- [ ] Add `clippy::pedantic` to CI (with allowlist)
- [ ] Set up automated code quality metrics
- [ ] Create contribution guidelines for code quality

## 🔧 Tools & Automation

### Run Clippy with Pedantic Checks
```bash
cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery
```

### Find Specific Issues
```bash
# Find unwrap calls
rg "\.unwrap\(\)" --type rust

# Find TODO comments
rg "TODO|FIXME" --type rust

# Find debug prints
rg "println!|dbg!|eprintln!" --type rust src/
```

### Auto-Fix Some Issues
```bash
cargo clippy --fix --allow-dirty --allow-staged
```

## 📝 Notes

- This is not a blocker for releases - all issues are quality improvements
- Focus on high-impact, low-effort fixes first
- Some "issues" are intentional design choices (document them)
- Balance perfectionism with pragmatism - working code > perfect code
