use std::ffi::{c_void, CString};

use mlir_sys::*;

pub struct MlirWrapper {
	pub optimization_level: i32,
	pub shared_library_paths: Vec<String>,
	pub context: MlirContext,
}

impl Drop for MlirWrapper {
	fn drop(&mut self) {
		unsafe {
			mlirContextDestroy(self.context);
		}
	}
}

impl Default for MlirWrapper {
	fn default() -> MlirWrapper {
		unsafe {
			let context = mlirContextCreate();

			let registry = mlirDialectRegistryCreate();

			mlirRegisterAllDialects(registry);
			mlirContextAppendDialectRegistry(context, registry);
			mlirDialectRegistryDestroy(registry);

			mlirContextGetOrLoadDialect(context, mlir_str("arith"));
			mlirContextGetOrLoadDialect(context, mlir_str("memref"));
			mlirContextGetOrLoadDialect(context, mlir_str("func"));

			MlirWrapper {
				context,
				shared_library_paths: vec![],
				optimization_level: 2,
			}
		}
	}
}

impl MlirWrapper {
	pub fn parse_module(&self, ir_code: &str) -> MlirModule {
		unsafe { mlirModuleCreateParse(self.context, mlir_str(ir_code)) }
	}

	pub fn create_module(&self) -> MlirModule {
		unsafe { mlirModuleCreateEmpty(mlirLocationUnknownGet(self.context)) }
	}

	///
	/// # Panics
	///
	/// Panics if there is a failure in the pass manager.
	///
	/// # Safety
	///
	pub fn lower_to_llvm(&self, module: MlirModule) {
		unsafe {
			mlirContextGetOrLoadDialect(self.context, mlir_str("llvm"));

			let pass_manager = mlirPassManagerCreate(self.context);
			let operation_pass_manager = mlirPassManagerGetNestedUnder(
				pass_manager,
				mlir_str("func.func"),
			);
			mlirPassManagerAddOwnedPass(
				pass_manager,
				mlirCreateConversionConvertFuncToLLVMPass(),
			);
			mlirOpPassManagerAddOwnedPass(
				operation_pass_manager,
				mlirCreateConversionArithToLLVMConversionPass(),
			);
			let status = mlirPassManagerRunOnOp(
				pass_manager,
				mlirModuleGetOperation(module),
			);
			mlirPassManagerDestroy(pass_manager);
			if status.value == 0 {
				// TODO: return error
				panic!("Lowering to LLVM failed");
			}

			mlirRegisterAllLLVMTranslations(self.context);
		}
	}

	pub fn execute(
		&self,
		module: MlirModule,
		func: &str,
		args: &mut [*mut c_void],
	) {
		unsafe {
			let shared_library_paths: Vec<MlirStringRef> = self
				.shared_library_paths
				.iter()
				.map(|path| mlir_str(path))
				.collect();

			let execution_engine = mlirExecutionEngineCreate(
				module,
				self.optimization_level,
				shared_library_paths.len() as i32,
				shared_library_paths.as_ptr(),
				false,
			);

			if execution_engine.ptr.is_null() {
				panic!("Failed to create an execution engine");
			}

			if mlirExecutionEngineInvokePacked(
				execution_engine,
				mlir_str(func),
				args.as_mut_ptr(),
			)
			.value == 0
			{
				panic!("Failed to invoke function {}", func);
			}

			mlirExecutionEngineDestroy(execution_engine);
			mlirModuleDestroy(module);
		}
	}
}

pub fn mlir_str(value: &str) -> MlirStringRef {
	let string = CString::new(value).unwrap();

	unsafe { mlirStringRefCreateFromCString(string.into_raw()) }
}
