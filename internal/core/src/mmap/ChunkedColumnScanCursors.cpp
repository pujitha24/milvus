// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#include "mmap/ChunkedColumnInterface.h"

namespace milvus {
namespace detail {

class PinnedScanInput final {
 public:
    PinnedScanInput(int64_t chunk_id, PinWrapper<Chunk*>&& chunk)
        : chunk_id_(chunk_id), chunk_(std::move(chunk)) {
    }

    Chunk*
    GetChunk(int64_t chunk_id) const {
        AssertInfo(chunk_id == chunk_id_,
                   "scan chunk {} is not pinned, current chunk {}",
                   chunk_id,
                   chunk_id_);
        return chunk_.get();
    }

 private:
    int64_t chunk_id_;
    PinWrapper<Chunk*> chunk_;
};

using StringViews = std::pair<std::vector<std::string_view>, FixedVector<bool>>;
using ArrayViews = std::pair<std::vector<ArrayView>, FixedVector<bool>>;
using VectorArrayViews =
    std::pair<std::vector<VectorArrayView>, FixedVector<bool>>;

struct StringOwner {
    StringOwner(std::shared_ptr<PinnedScanInput> input, StringViews&& views)
        : input(std::move(input)), views(std::move(views)) {
    }

    std::shared_ptr<PinnedScanInput> input;
    StringViews views;
};

struct JsonOwner {
    JsonOwner(std::shared_ptr<PinnedScanInput> input, StringViews&& views)
        : input(std::move(input)), views(std::move(views)) {
        values.reserve(this->views.first.size());
        for (const auto& value : this->views.first) {
            values.emplace_back(value);
        }
    }

    std::shared_ptr<PinnedScanInput> input;
    StringViews views;
    std::vector<Json> values;
};

struct ArrayOwner {
    ArrayOwner(std::shared_ptr<PinnedScanInput> input, ArrayViews&& views)
        : input(std::move(input)), views(std::move(views)) {
    }

    std::shared_ptr<PinnedScanInput> input;
    ArrayViews views;
};

struct VectorArrayOwner {
    VectorArrayOwner(std::shared_ptr<PinnedScanInput> input,
                     VectorArrayViews&& views)
        : input(std::move(input)), views(std::move(views)) {
    }

    std::shared_ptr<PinnedScanInput> input;
    VectorArrayViews views;
};

class RawScanCursor final : public ChunkedColumnInterface::ScanCursor {
 public:
    RawScanCursor(const ChunkedColumnInterface* column,
                  milvus::OpContext* op_ctx,
                  int64_t start_offset,
                  DataType data_type,
                  ChunkedColumnInterface::ScanValueKind value_kind,
                  ChunkedColumnInterface::ScanPinPolicy pin_policy)
        : column_(column),
          op_ctx_(op_ctx),
          data_type_(data_type),
          value_kind_(value_kind),
          pin_policy_(pin_policy),
          scan_pos_(start_offset) {
        AssertInfo(column_ != nullptr, "raw scan column is null");
        AssertInfo(start_offset >= 0 &&
                       start_offset <= static_cast<int64_t>(column_->NumRows()),
                   "data scan start {} out of rows {}",
                   start_offset,
                   column_->NumRows());
        SetPhysicalPosition(start_offset);
    }

    bool
    Next(int64_t position,
         int64_t length,
         ChunkedColumnInterface::ScanReadMode mode,
         ChunkedColumnInterface::ScanBatch* out) override {
        AssertInfo(out != nullptr, "raw scan output batch is null");
        ResetOutput(out);
        AssertInfo(position >= scan_pos_,
                   "raw scan cannot seek backward from {} to {}",
                   scan_pos_,
                   position);
        AssertInfo(
            length >= 0 &&
                position <= static_cast<int64_t>(column_->NumRows()) &&
                length <= static_cast<int64_t>(column_->NumRows()) - position,
            "raw scan range [{}, {}) out of rows {}",
            position,
            position + length,
            column_->NumRows());
        if (position != scan_pos_) {
            Seek(position);
        }
        if (length == 0 ||
            scan_pos_ == static_cast<int64_t>(column_->NumRows())) {
            return false;
        }
        AssertInfo(mode != ChunkedColumnInterface::ScanReadMode::ValidityOnly ||
                       column_->IsNullable(),
                   "validity-only scan requested for non-nullable column");

        const auto chunk_id = current_chunk_id_;
        const auto chunk_rows = column_->chunk_row_nums(chunk_id);
        const auto size =
            std::min<int64_t>(length, chunk_rows - current_chunk_offset_);
        AssertInfo(size > 0,
                   "raw scan made no progress in chunk {} at offset {}",
                   chunk_id,
                   current_chunk_offset_);

        auto input = PinChunk(chunk_id);
        auto* chunk = input->GetChunk(chunk_id);
        out->row_id_start = scan_pos_;
        out->size = size;

        if (mode == ChunkedColumnInterface::ScanReadMode::ValidityOnly) {
            FillValidityOnly(input, chunk, out);
        } else if (value_kind_ ==
                   ChunkedColumnInterface::ScanValueKind::FixedWidth) {
            FillFixedWidth(input, chunk, out);
        } else {
            FillView(input, chunk, out);
        }

        Advance(size);
        UpdateStickyInput(chunk_id, input);
        return true;
    }

 private:
    static void
    ResetOutput(ChunkedColumnInterface::ScanBatch* out) {
        out->values = ChunkedColumnInterface::ValueView{};
        out->validity = nullptr;
        out->row_ids.clear();
        out->owner.reset();
        out->row_id_start = 0;
        out->size = 0;
    }

    void
    SetPhysicalPosition(int64_t position) {
        scan_pos_ = position;
        if (position == static_cast<int64_t>(column_->NumRows())) {
            current_chunk_id_ = column_->num_chunks();
            current_chunk_offset_ = 0;
            return;
        }
        const auto [chunk_id, chunk_offset] =
            column_->GetChunkIDByOffset(position);
        current_chunk_id_ = static_cast<int64_t>(chunk_id);
        current_chunk_offset_ = static_cast<int64_t>(chunk_offset);
    }

    void
    Seek(int64_t position) {
        SetPhysicalPosition(position);
        if (sticky_input_ != nullptr && sticky_chunk_id_ != current_chunk_id_) {
            sticky_input_.reset();
            sticky_chunk_id_ = -1;
        }
    }

    std::shared_ptr<PinnedScanInput>
    PinChunk(int64_t chunk_id) {
        if (sticky_input_ != nullptr && sticky_chunk_id_ == chunk_id) {
            return sticky_input_;
        }
        return std::make_shared<PinnedScanInput>(
            chunk_id, column_->GetChunk(op_ctx_, chunk_id));
    }

    void
    FillValidityOnly(const std::shared_ptr<PinnedScanInput>& input,
                     Chunk* chunk,
                     ChunkedColumnInterface::ScanBatch* out) const {
        const auto& valid = chunk->Valid();
        AssertInfo(current_chunk_offset_ + out->size <=
                       static_cast<int64_t>(valid.size()),
                   "raw validity range [{}, {}) out of chunk validity {}",
                   current_chunk_offset_,
                   current_chunk_offset_ + out->size,
                   valid.size());
        out->validity = valid.data() + current_chunk_offset_;
        out->owner = input;
    }

    void
    FillFixedWidth(const std::shared_ptr<PinnedScanInput>& input,
                   Chunk* chunk,
                   ChunkedColumnInterface::ScanBatch* out) const {
        auto* fixed_chunk = dynamic_cast<FixedWidthChunk*>(chunk);
        AssertInfo(fixed_chunk != nullptr,
                   "scan chunk {} is not fixed-width",
                   current_chunk_id_);
        const auto span = fixed_chunk->Span();
        AssertInfo(
            span.row_count() == column_->chunk_row_nums(current_chunk_id_),
            "scan chunk {} row count mismatch, metadata {}, span {}",
            current_chunk_id_,
            column_->chunk_row_nums(current_chunk_id_),
            span.row_count());
        out->values.encoding =
            ChunkedColumnInterface::ValueEncoding::FixedWidth;
        out->values.kind = value_kind_;
        out->values.physical_type = data_type_;
        out->values.logical_type = data_type_;
        out->values.data = span.data();
        out->values.offset = current_chunk_offset_;
        out->values.size = out->size;
        out->values.byte_width = span.element_sizeof();
        if (span.valid_data() != nullptr) {
            out->validity = span.valid_data() + current_chunk_offset_;
        }
        out->owner = input;
    }

    void
    FillView(const std::shared_ptr<PinnedScanInput>& input,
             Chunk* chunk,
             ChunkedColumnInterface::ScanBatch* out) const {
        const auto range = std::make_pair(current_chunk_offset_, out->size);
        switch (value_kind_) {
            case ChunkedColumnInterface::ScanValueKind::StringView: {
                auto* string_chunk = dynamic_cast<StringChunk*>(chunk);
                AssertInfo(string_chunk != nullptr,
                           "scan chunk {} is not string-like",
                           current_chunk_id_);
                auto owner = std::make_shared<StringOwner>(
                    input, string_chunk->StringViews(range));
                out->values.encoding =
                    ChunkedColumnInterface::ValueEncoding::StringView;
                out->values.kind = value_kind_;
                out->values.physical_type = data_type_;
                out->values.logical_type = data_type_;
                out->values.data = owner->views.first.data();
                out->values.offset = 0;
                out->values.size = out->size;
                out->values.byte_width = sizeof(std::string_view);
                FillViewValidity(owner->views.second, out);
                out->owner = std::move(owner);
                return;
            }
            case ChunkedColumnInterface::ScanValueKind::JsonView: {
                auto* string_chunk = dynamic_cast<StringChunk*>(chunk);
                AssertInfo(string_chunk != nullptr,
                           "scan chunk {} is not JSON",
                           current_chunk_id_);
                auto owner = std::make_shared<JsonOwner>(
                    input, string_chunk->StringViews(range));
                out->values.encoding =
                    ChunkedColumnInterface::ValueEncoding::JsonView;
                out->values.kind = value_kind_;
                out->values.physical_type = data_type_;
                out->values.logical_type = DataType::JSON;
                out->values.data = owner->values.data();
                out->values.offset = 0;
                out->values.size = out->size;
                out->values.byte_width = sizeof(Json);
                FillViewValidity(owner->views.second, out);
                out->owner = std::move(owner);
                return;
            }
            case ChunkedColumnInterface::ScanValueKind::ArrayView: {
                auto* array_chunk = dynamic_cast<ArrayChunk*>(chunk);
                AssertInfo(array_chunk != nullptr,
                           "scan chunk {} is not an array",
                           current_chunk_id_);
                auto owner = std::make_shared<ArrayOwner>(
                    input, array_chunk->Views(range));
                out->values.encoding =
                    ChunkedColumnInterface::ValueEncoding::ArrayView;
                out->values.kind = value_kind_;
                out->values.physical_type = data_type_;
                out->values.logical_type = DataType::ARRAY;
                out->values.data = owner->views.first.data();
                out->values.offset = 0;
                out->values.size = out->size;
                out->values.byte_width = sizeof(ArrayView);
                FillViewValidity(owner->views.second, out);
                out->owner = std::move(owner);
                return;
            }
            case ChunkedColumnInterface::ScanValueKind::VectorArrayView: {
                auto* array_chunk = dynamic_cast<VectorArrayChunk*>(chunk);
                AssertInfo(array_chunk != nullptr,
                           "scan chunk {} is not a vector array",
                           current_chunk_id_);
                auto owner = std::make_shared<VectorArrayOwner>(
                    input, array_chunk->Views(range));
                out->values.encoding =
                    ChunkedColumnInterface::ValueEncoding::VectorArrayView;
                out->values.kind = value_kind_;
                out->values.physical_type = data_type_;
                out->values.logical_type = DataType::VECTOR_ARRAY;
                out->values.data = owner->views.first.data();
                out->values.offset = 0;
                out->values.size = out->size;
                out->values.byte_width = sizeof(VectorArrayView);
                FillViewValidity(owner->views.second, out);
                out->owner = std::move(owner);
                return;
            }
            default:
                ThrowInfo(ErrorCode::Unsupported,
                          "unsupported raw scan value kind {}",
                          static_cast<int>(value_kind_));
        }
    }

    void
    FillViewValidity(const FixedVector<bool>& valid,
                     ChunkedColumnInterface::ScanBatch* out) const {
        if (column_->IsNullable() && !valid.empty()) {
            out->validity = valid.data();
        }
    }

    void
    Advance(int64_t size) {
        scan_pos_ += size;
        current_chunk_offset_ += size;
        const auto chunk_rows = column_->chunk_row_nums(current_chunk_id_);
        if (current_chunk_offset_ == chunk_rows) {
            if (scan_pos_ < static_cast<int64_t>(column_->NumRows())) {
                ++current_chunk_id_;
                current_chunk_offset_ = 0;
            } else {
                current_chunk_id_ = column_->num_chunks();
                current_chunk_offset_ = 0;
            }
        }
    }

    void
    UpdateStickyInput(int64_t returned_chunk_id,
                      const std::shared_ptr<PinnedScanInput>& input) {
        if (pin_policy_ !=
                ChunkedColumnInterface::ScanPinPolicy::UntilCellExhausted ||
            scan_pos_ == static_cast<int64_t>(column_->NumRows()) ||
            current_chunk_id_ != returned_chunk_id) {
            sticky_input_.reset();
            sticky_chunk_id_ = -1;
            return;
        }
        sticky_input_ = input;
        sticky_chunk_id_ = returned_chunk_id;
    }

    const ChunkedColumnInterface* column_;
    milvus::OpContext* op_ctx_;
    DataType data_type_;
    ChunkedColumnInterface::ScanValueKind value_kind_;
    ChunkedColumnInterface::ScanPinPolicy pin_policy_;
    int64_t scan_pos_{0};
    int64_t current_chunk_id_{0};
    int64_t current_chunk_offset_{0};
    std::shared_ptr<PinnedScanInput> sticky_input_;
    int64_t sticky_chunk_id_{-1};
};

inline ChunkedColumnInterface::ScanResult
OpenDataScan(const ChunkedColumnInterface* column,
             milvus::OpContext* op_ctx,
             int64_t start_offset,
             DataType data_type,
             ChunkedColumnInterface::ScanValueKind value_kind,
             ChunkedColumnInterface::ScanPinPolicy pin_policy) {
    const auto column_kind = GetScanValueKindForDataType(data_type);
    if (!column_kind.has_value()) {
        return nullptr;
    }

    const auto resolved_kind =
        value_kind == ChunkedColumnInterface::ScanValueKind::Default
            ? *column_kind
            : value_kind;
    AssertInfo(resolved_kind == *column_kind,
               "data scan kind {} does not match column type {}, expected {}",
               static_cast<int>(resolved_kind),
               data_type,
               static_cast<int>(*column_kind));
    return std::make_unique<RawScanCursor>(
        column, op_ctx, start_offset, data_type, resolved_kind, pin_policy);
}

}  // namespace detail

ChunkedColumnInterface::ScanResult
ChunkedColumnInterface::Scan(milvus::OpContext* op_ctx,
                             const ScanOptions& options) const {
    if (options.output != ScanOutput::Data ||
        options.predicate != ScanPredicate::None) {
        return nullptr;
    }
    auto data_type = GetDefaultScanDataType();
    if (!data_type.has_value()) {
        return nullptr;
    }
    return detail::OpenDataScan(this,
                                op_ctx,
                                options.start_offset,
                                *data_type,
                                options.value_kind,
                                options.pin_policy);
}

}  // namespace milvus
