use std::ffi::c_void;

use mlir_sys::*;
use tsmlir::{mlir_str, MlirWrapper};

fn main() {
	unsafe {
		let wrapper = MlirWrapper::default();

		let module = wrapper.create_module();

		let memref_type =
			mlirTypeParseGet(wrapper.context, mlir_str("memref<?xf32>"));

		let location = mlirLocationUnknownGet(wrapper.context);
		let module_body = mlirModuleGetBody(module);
		let func_body = mlirBlockCreate(
			2,
			[memref_type, memref_type].as_ptr(),
			[location, location].as_ptr(),
		);
		let func_body_region = mlirRegionCreate();

		mlirRegionAppendOwnedBlock(func_body_region, func_body);

		let func_type_attr = mlirAttributeParseGet(
			wrapper.context,
			mlir_str("(memref<?xf32>, memref<?xf32>) -> ()"),
		);

		let func_name_attr =
			mlirAttributeParseGet(wrapper.context, mlir_str("\"add\""));

		let mut func_attrs = vec![
			mlirNamedAttributeGet(
				mlirIdentifierGet(wrapper.context, mlir_str("function_type")),
				func_type_attr,
			),
			mlirNamedAttributeGet(
				mlirIdentifierGet(wrapper.context, mlir_str("sym_name")),
				func_name_attr,
			),
		];

		let mut operation_state =
			mlirOperationStateGet(mlir_str("func.func"), location);

		mlirOperationStateAddAttributes(
			&mut operation_state,
			2,
			func_attrs.as_mut_ptr(),
		);
		mlirOperationStateAddOwnedRegions(
			&mut operation_state,
			1,
			&func_body_region,
		);
		let func = mlirOperationCreate(&mut operation_state);
		mlirBlockInsertOwnedOperation(module_body, 0, func);

		// TODO: parse and convert ecmascript AST

		wrapper.lower_to_llvm(module);

		let mut input = 42_i32;
		let mut result = -1_i32;

		wrapper.execute(
			module,
			"add",
			&mut [
				&mut input as *mut _ as *mut c_void,
				&mut result as *mut _ as *mut c_void,
			],
		);

		dbg!(result);
	}
}
